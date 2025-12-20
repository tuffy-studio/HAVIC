#---------------------------------------------------------------------------
# Copyright (c) Jielun Peng, Harbin Institute of Technology, 2025.
# All rights reserved.
#---------------------------------------------------------------------------

# ================= NOTE: the input and output ================
# input: a csv file containing: (video_path, audio label, visual_label, overall_label) or (video_path, overall_label)
# for labels, 1 means fake, 0 means real

# output1: the statstics of the inference results, including ACC, AUC, AP, T1, T0.1. 
# output2: a csv file containing: (video_path, audio_label, audio pred, visual_label, visual_pred, overall_label, overall_pred) or (video_path, label, pred)
# =============================================================

# --------------------------- import module / package ----------------------------------
import argparse
import os
import csv
from tqdm import tqdm

import numpy as np
import cv2

import sys
 # 获取当前脚本所在的目录，添加环境变量
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
        help="Input CSV file path"
    )
    parser.add_argument(
        "--save_csv_path",
        type=str,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        help="Path to pretrained model"
    )

    # =========================== sliding window hyperparameters ===========================
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--window_stride_frames", type=int, default=2)
    
    parser.add_argument("--window_size_fbank", type=int, default=1024)
    parser.add_argument("--window_stride_fbank", type=int, default=128)

    parser.add_argument("--max_time", type=int, default=10, help="Max processing time (seconds)")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of workers")

    parser.add_argument(
        "--classify_loss",
        type=str,
        default="BCE",
        choices=["BCE", "CE"],
        help="Classification loss type"
    )

    return parser.parse_args()

args = parse_args()
csv_file_path = args.csv_file_path
save_csv_path = args.save_csv_path
if not os.path.exists(save_csv_path):
    with open(save_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "overall_label", "pred_score"])

pretrain_path = args.pretrain_path
classify_loss = args.classify_loss

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

    duration = total_frames / fps  # 视频总时长（秒）

    # 每0.2s采样一个时间点
    times = np.arange(0, duration, 0.2)
    num_sampled_frames = len(times)
    # 转换为对应帧号
    sample_indices = (times * fps).astype(int)
    sample_indices = np.clip(sample_indices, 0, total_frames - 1)  # 防止越界
    sample_indices = np.unique(sample_indices)  # 去重

    # 检查是否已存在相同数量的采样帧
    existing_frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    if len(existing_frames) >= len(sample_indices):
        #print(f"Frames already exist, skipping: {frame_dir}")
        return num_sampled_frames

    frames = vr.get_batch(sample_indices).asnumpy()
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frame_dir, f"frame_{idx:02d}.png")
        if not os.path.exists(frame_path):  # 避免覆盖
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            #imageio.imwrite(frame_path, frame)
        if idx > MAX_TIME*5:
            break
    
    return num_sampled_frames

def load_face_detection_model():
    # 生成绝对路径，确保无论在哪个目录调用，路径都正确
    config_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'config', 'model_conf.yaml')
    model_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'models')

    with open(config_path, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.SafeLoader)

    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]

    # 加载模型
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
        从单张图像中提取第一个人脸区域并保存
        """
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"无法读取图像: {frame_path}")
        
        dets = self.faceDetModelHandler.inference_on_image(frame)
        if len(dets) == 0:
            raise ValueError("未检测到人脸")
            return False

        x_min, y_min, x_max, y_max = map(int, dets[0][:4])
        H, W = frame.shape[:2]

        # 缩放人脸框
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
    """从.mp4文件中提取音频并保存为.wav格式"""
    if os.path.exists(output_audio_file):
        return

    try:
        # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
        ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(
            quiet=True)  # ac=1 表示单声道，ar='16k' 设置采样率
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

state_dict = torch.load(pretrain_path, map_location='cpu')
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
def process_single_video(index, row, classify_loss = "BCE"):
    results = {}
    try:
        video_path = row.get("video_path", row.get("video_name"))
        overall_label = row["overall_label"]

        # ---------------- 2.3.1 video preprocess ----------------
        parts = video_path.split(os.sep)
        video_name = os.path.splitext("_".join(parts[-3:]))[0]

        # i.extract frame from video
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
                        tag = face_cropper.extract(frame_path, save_path)
                        if tag == False:
                            continue
                    frame = Image.open(save_path).convert('RGB')
                    frame = tensor_process(frame)
                    face_frames_list.append(frame)
                    count += 1
                    if count >= MAX_TIME*5:
                        break
                except Exception as e:
                    print(f"[Warning] {video_name}: Failed to process {frame_file}: {e}")
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

        #total_audio_len = waveform.size(1)
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
        print(f"{video_name}: video_batch {video_batch.shape}, audio_batch {audio_batch.shape}")

        # ---------------- 2.3.4 model forward ----------------
        with torch.inference_mode(), torch.autocast("cuda"):
            audio_outputs, visual_outputs, output = model(audio=audio_batch, video=video_batch)
        sample_level_pred = output.detach().cpu().mean(dim=0)

        results["pred"] = sample_level_pred.unsqueeze(0)
        if classify_loss == "CE":
            results["label"] = torch.tensor([int(overall_label)], dtype=torch.long)
        else:
            results["label"] = torch.tensor([int(overall_label)], dtype=torch.float32)
        
        results["video_name"] = video_name
        print(f"[{index+1}/{len(data)}] {video_name}: label {overall_label}, pred {sample_level_pred.numpy()}", flush=True)
        return results

    except Exception as e:
        print(f"[Error] {row['video_name']}: {e}")
        return None

# ---------------- 3. 多线程执行 ----------------
csv_records = []
all_overall_pred, all_overall_label = [], []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 改成你机器的CPU核心数
    futures = [executor.submit(process_single_video, i, row, classify_loss) for i, row in enumerate(data)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Sliding Window Inference"):
        res = future.result()
        if res is not None:
            # res 包含: {"video_name": str, "pred": tensor, "label": tensor}
            video_name = res["video_name"]
            pred_score = res["pred"].item() if torch.is_tensor(res["pred"]) else float(res["pred"])
            overall_label = int(res["label"]) if not torch.is_tensor(res["label"]) else int(res["label"].item())

            # 保存到列表
            all_overall_pred.append(pred_score)
            all_overall_label.append(overall_label)

            # 记录到 CSV
            csv_records.append({
                "video_name": video_name,
                "overall_label": overall_label,
                "pred_score": pred_score
            })

            with open(save_csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["video_name", "overall_label", "pred_score"])
                writer.writeheader()
                writer.writerows(csv_records)

print(f"推理结果已保存到: {save_csv_path}")

# ---------------- 4. 计算指标 ----------------
overall_output = torch.tensor(all_overall_pred, dtype=torch.float32)  # [N,1]
overall_label  = torch.tensor(all_overall_label, dtype=torch.float32) # [N,1]


if classify_loss == "CE":
    plot_classwise_logits_histogram(overall_output, overall_label, normalize=True,
        save_path="./sliding_window_inference_logs/classwise_logits.png"
    )
    one_hot_target = F.one_hot(overall_label, num_classes=2).float()
    stats = calculate_stats(overall_output.cpu(), one_hot_target.cpu())
else:
    plot_classwise_logits_histogram_bce(
        overall_output, overall_label, normalize=False,
        save_path="./sliding_window_inference_logs/classwise_logits.png"
    )
    overall_label = overall_label.unsqueeze(1)
    stats = calculate_stats(torch.sigmoid(overall_output).cpu(), overall_label.cpu())

for i, s in enumerate(stats):
    print(f"Class {i} - ACC: {s['ACC']:.4f}, AP: {s['AP']:.4f}, AUC: {s['AUC']:.4f}, T1: {s['T1']:.4f}, T0.1: {s['T0.1']:.4f}, f1: {s['F1']:.4f}")



