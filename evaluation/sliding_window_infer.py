#---------------------------------------------------------------------------
# Copyright (c) Jielun Peng, Harbin Institute of Technology, 2025.
# All rights reserved.
#---------------------------------------------------------------------------

# ================= NOTE: the input and output ================
# input: a csv file containing: (video_path, audio label, visual_label, overall_label) or (video_path, overall_label)
# for labels, 1 means fake, 0 means real

# output1: a csv file containing: (video_path, overall_label, overall_pred) for evaluation, (video_path, overall_pred) for inference
# output2 (if the mode is evaluation): the statstics of the evaluation results, including ACC, AP, AUC 

# =============================================================

# --------------------------- import module / package ----------------------------------
import argparse
import os
import csv
from tqdm import tqdm

import numpy as np
import cv2

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk"))
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk/models/network_def"))
import yaml

from FaceX_Zoo.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from FaceX_Zoo.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stats import *
from src.models.HAVIC import HAVIC_FT

import ffmpeg
import torchaudio

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------

# ------------------------ Setting the configuration ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sliding Window Inference Config")

    # =========================== paths ===========================
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--save_csv_path",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--finetune_path",
        type=str,
        required=True,
        help="Path to finetuned model"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="evaluation or inference mode"
    )

    # =========================== sliding window hyperparameters ===========================
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--window_stride_frames", type=int, default=2)
    
    parser.add_argument("--window_size_fbank", type=int, default=1024)
    parser.add_argument("--window_stride_fbank", type=int, default=128)

    parser.add_argument("--max_time", type=int, default=10, help="Max processing time (seconds)")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of workers")

    return parser.parse_args()

args = parse_args()
csv_file_path = args.csv_file_path
save_csv_path = args.save_csv_path
mode = args.mode
finetune_path = args.finetune_path

# Hyperparameters for sliding window
WINDOW_SIZE_FRAMES = args.window_size_frames       # number of frames per window
WINDOW_STRIDE_FRAMES = args.window_stride_frames      # stride in frames
WINDOW_SIZE_FBANK = args.window_size_fbank      # corresponding audio length per window
WINDOW_STRIDE_FBANK = args.window_stride_fbank     # corresponding audio stride

MAX_TIME = args.max_time # seconds, max time to process for each video
MAX_WORKERS = args.max_workers
# ---------------------------------------------------------------------------


#------------------- audio/visual preprocess function ----------------------
def sample_video_5fps_decord(video_path, frame_output_root='./sliding_window_inference_tmp/sampled_frames'):

    os.makedirs(frame_output_root, exist_ok=True)

    frame_dir = frame_output_root
    os.makedirs(frame_dir, exist_ok=True)

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"Failed to open video: {video_path}, error: {e}")
        return -1

    total_frames = len(vr)
    if total_frames < 1:
        print(f"Video has fewer than 1 frame, skipping: {video_path}")
        return -1

    fps = vr.get_avg_fps()
    if fps is None or fps <= 0:
        print(f"Failed to retrieve FPS, skipping: {video_path}")
        return -1

    duration = total_frames / fps  # Total video duration (seconds)

    # Sample one time point every 0.2 seconds
    times = np.arange(0, duration, 0.2)
    num_sampled_frames = len(times)
    # Convert to corresponding frame indices
    sample_indices = (times * fps).astype(int)
    sample_indices = np.clip(sample_indices, 0, total_frames - 1) # Prevent out-of-bounds
    sample_indices = np.unique(sample_indices)  # Remove duplicates

    # Check if the sampled frames already exists
    existing_frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    if len(existing_frames) >= len(sample_indices):
        #print(f"Frames already exist, skipping: {frame_dir}")
        return num_sampled_frames

    frames = vr.get_batch(sample_indices).asnumpy()
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frame_dir, f"frame_{idx:02d}.png")
        if not os.path.exists(frame_path):  # Avoid overwriting if the file already exists
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if idx > MAX_TIME*5:
            break
    
    return num_sampled_frames

def load_face_detection_model():
    config_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'config', 'model_conf.yaml')
    model_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'models')

    with open(config_path, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.SafeLoader)

    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]

    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name, meta_file="model_meta.json")
    model, cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:5', cfg)
    
    if faceDetModelHandler is not None and faceDetModelLoader is not None:
        print("Face detection model loaded successfully.")
    return faceDetModelHandler

class FaceX_Zoo_FaceCropper:
    def __init__(self, faceDetModelHandler, scale=1.3):
        self.faceDetModelHandler = faceDetModelHandler
        self.scale = scale

    def extract(self, frame_path, save_path):
        """
        Extract the face region from a single image and save it
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"can not read image: {frame_path}")

        dets = self.faceDetModelHandler.inference_on_image(frame)
        if len(dets) == 0:
            raise ValueError(f"fail to detect face: {frame_path}")

        x_min, y_min, x_max, y_max = map(int, dets[0][:4])
        H, W = frame.shape[:2]

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        new_w = (x_max - x_min) * self.scale
        new_h = (y_max - y_min) * self.scale
        x_min = max(int(cx - new_w / 2), 0)
        x_max = min(int(cx + new_w / 2), W)
        y_min = max(int(cy - new_h / 2), 0)
        y_max = min(int(cy + new_h / 2), H)

        face_crop = frame[y_min:y_max, x_min:x_max]
        Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).save(save_path)

        return True

def extract_audio_from_video(video_file, output_audio_file):
    """Extract audio from a .mp4 file and save it as a .wav file"""
    if os.path.exists(output_audio_file):
        return

    try:
        ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(
            quiet=True)  
        print(f"Audio successfully extracted to {output_audio_file}.")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e}")

#---------------------------------------------------------------------------

# ====================== NOTE: the main code =====================

# ---------------- 1. reading the csv file ----------------
data = []
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)
print(f'According to the csv file, there are {len(data)} samples.')

# ---------------- 2. Sliding Window Inference ----------------

# 2.1 load detection model
model = HAVIC_FT()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(finetune_path, map_location='cpu')
if not isinstance(model, nn.DataParallel):
    model = nn.DataParallel(model)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("[Load Weights Info] Missing keys:", missing_keys)
print("[Load Weights Info] Unexpected keys:", unexpected_keys)
model.eval().to(device)

# 2.2 load face_cropper model
face_detection_model = load_face_detection_model()
face_cropper = FaceX_Zoo_FaceCropper(faceDetModelHandler=face_detection_model, scale=1.3)

from decord import VideoReader
from decord import cpu

# ---------------- helper: 单个视频推理 ----------------
def process_single_video(index, row, mode):
    results = {}
    try:
        video_path = row.get("video_path")

        if mode == "evaluation":
            overall_label = row["overall_label"]

        # ---------------- 2.3.1 video preprocess ----------------
        # i.extract frame from video

        # Convert the full video path into a valid folder name for saving temporary files
        # This avoids issues caused by '/' or '\' in the path
        video_name = video_path.replace(os.sep, "_") 
        frame_output_dir = f"./sliding_window_inference_tmp/sampled_frames/{video_name}"
        os.makedirs(frame_output_dir, exist_ok=True)

        total_frames = sample_video_5fps_decord(video_path, frame_output_dir)

        if total_frames == -1:
            return None

        # ii.extract face from frame
        cropped_face_dir = f"./sliding_window_inference_tmp/cropped_faces/{video_name}"
        os.makedirs(cropped_face_dir, exist_ok=True)

        tensor_process = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
        ])

        face_frames_list = []
        count = 0
        for frame_file in sorted(os.listdir(frame_output_dir)):
            if frame_file.lower().endswith((".png", ".jpg", ".jpeg")):
                frame_path = os.path.join(frame_output_dir, frame_file)
                save_path = os.path.join(cropped_face_dir, frame_file)
                try:
                    if not os.path.exists(save_path):
                        face_cropper.extract(frame_path, save_path)
                    frame = Image.open(save_path).convert('RGB')
                    frame = tensor_process(frame)
                    face_frames_list.append(frame)
                    count += 1
                    if count >= MAX_TIME*5:
                        break
                except Exception as e:
                    print(f"[Warning] {video_path}: Failed to process {frame_file}: {e}")
                    return None

        if len(face_frames_list) < 16:
            return None

        remainder = (len(face_frames_list)-WINDOW_SIZE_FRAMES) % WINDOW_STRIDE_FRAMES
        if remainder != 0:
            face_frames_list = face_frames_list[:-remainder]

        frames_tensor = torch.stack(face_frames_list).to(device)
        sliding_windows_num = (len(face_frames_list) - WINDOW_SIZE_FRAMES)//WINDOW_STRIDE_FRAMES + 1

        # ---------------- 2.3.2 audio preprocess ----------------
        audio_wav_file = f"./sliding_window_inference_tmp/audio_wav/{video_name}.wav"
        os.makedirs(os.path.dirname(audio_wav_file), exist_ok=True)
        extract_audio_from_video(video_path, audio_wav_file)

        waveform, sr = torchaudio.load(audio_wav_file)

        # ---------------- 2.3.3 create sliding window batches ----------------
        video_windows = []
        for start in range(0, frames_tensor.size(0) - WINDOW_SIZE_FRAMES + 1, WINDOW_STRIDE_FRAMES):
            end = start + WINDOW_SIZE_FRAMES
            video_windows.append(frames_tensor[start:end].permute(1, 0, 2, 3))    
        video_batch = torch.stack(video_windows)  # [num_windows, 3, T, H, W]

        total_audio_len = int((len(face_frames_list) / total_frames) * waveform.size(1))
        segment_len = int((total_audio_len*WINDOW_SIZE_FRAMES)/(WINDOW_SIZE_FRAMES + WINDOW_STRIDE_FRAMES*(sliding_windows_num-1)))
        stride = int((WINDOW_STRIDE_FRAMES/WINDOW_SIZE_FRAMES)*segment_len)

        audio_segments = []
        for i in range(sliding_windows_num):
            start = i * stride
            end = start + segment_len
            if end > total_audio_len:
                end = total_audio_len
                start = max(0, end - segment_len)
            segment = waveform[:, int(start):int(end)]
            segment = segment - segment.mean()
            audio_segments.append(segment)

        fbank_list = []
        for seg in audio_segments:
            fbank = torchaudio.compliance.kaldi.fbank(
                seg, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
            )
            fbank = torch.nn.functional.interpolate(
                fbank.unsqueeze(0).transpose(1, 2), size=(1024,),
                mode='linear', align_corners=False).transpose(1, 2).squeeze(0).to(device)
            fbank = (fbank - (-6.9960)) / (3.1205)
            fbank_list.append(fbank)

        audio_batch = torch.stack([f.to(device) for f in fbank_list])
        assert audio_batch.size(0) == video_batch.size(0), "batch size mismatch!"
        print(f"{video_path}:\n video_batch {video_batch.shape}, audio_batch {audio_batch.shape}")

        # ---------------- 2.3.4 model forward ----------------
        with torch.inference_mode(), torch.autocast("cuda"):
            audio_outputs, visual_outputs, output = model(audio=audio_batch, video=video_batch)
            sigmoid_output = torch.sigmoid(output)
        sample_level_pred = sigmoid_output.detach().cpu().mean(dim=0)

        results["overall_pred"] = sample_level_pred.unsqueeze(0)

        if mode == "evaluation":
                results["overall_label"] = torch.tensor([int(overall_label)], dtype=torch.float32)
        results["video_path"] = video_path

        if mode == "evaluation":
            print(f"[{index+1}/{len(data)}] {video_path}: overall_label {overall_label}, overall_pred {sample_level_pred.numpy()}", flush=True)
        else:
            print(f"[{index+1}/{len(data)}] {video_path}: overall_pred {sample_level_pred.numpy()}", flush=True)
        
        return results

    except Exception as e:
        print(f"[Error] {row['video_path']}: {e}")
        return None

# ---------------- 3. 多线程执行 ----------------
csv_records = []
all_overall_pred, all_overall_label = [], []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 改成你机器的CPU核心数
    futures = [executor.submit(process_single_video, i, row, mode) for i, row in enumerate(data)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Sliding Window Inference"):
        res = future.result()
        if res is not None:
            # res: {"video_path": str,  "overall_label": tensor, "overall_pred": tensor}
            video_path = res["video_path"]
            overall_pred = res["overall_pred"].item() if torch.is_tensor(res["overall_pred"]) else float(res["overall_pred"])
            all_overall_pred.append(overall_pred)

            if mode == "evaluation":
                overall_label = int(res["overall_label"]) if not torch.is_tensor(res["overall_label"]) else int(res["overall_label"].item())
                all_overall_label.append(overall_label)

            
            # write into CSV
            if mode == "evaluation":
                csv_records.append({
                    "video_path": video_path,
                    "overall_label": overall_label,
                    "overall_pred": overall_pred
                })
            else:
                csv_records.append({
                    "video_path": video_path,
                    "overall_pred": overall_pred
                })

    with open(save_csv_path, mode="w", newline="") as f:
        if mode=="evaluation":
            fieldnames = ["video_path", "overall_label", "overall_pred"]
        else:
            fieldnames = ["video_path", "overall_pred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_records)

print(f"Inference results have been saved to: {save_csv_path}")


# ---------------- 4. caculate the statstics of the evaluation results ----------------
if mode == "evaluation":
    overall_output = torch.tensor(all_overall_pred, dtype=torch.float32)  # [N,1]
    overall_label  = torch.tensor(all_overall_label, dtype=torch.float32) # [N,1]

    plot_classwise_logits_histogram_bce(
        overall_output, overall_label, normalize=False,
        save_path="./sliding_window_inference_logs/classwise_logits.png"
    )
    overall_label = overall_label.unsqueeze(1)
    stats = calculate_stats(torch.sigmoid(overall_output).cpu(), overall_label.cpu())

    for i, s in enumerate(stats):
        print(f"ACC: {s['ACC']:.4f}, AP: {s['AP']:.4f}, AUC: {s['AUC']:.4f}, T1: {s['T1']:.4f}, T0.1: {s['T0.1']:.4f}, f1: {s['F1']:.4f}")



