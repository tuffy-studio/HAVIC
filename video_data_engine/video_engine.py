# Copyright (c) Jielun Peng, Harbin Institute of Technology.
# All rights reserved.
# Reference: https://github.com/JDAI-CV/faceX-Zoo

import faulthandler
faulthandler.enable()
import cv2
cv2.setNumThreads(0)
import csv
from tqdm import tqdm
import os
from moviepy.editor import VideoFileClip
from decord import VideoReader
from decord import cpu
import imageio
import numpy as np
from PIL import Image
import math

import sys
 # 获取当前脚本所在的目录，添加环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk"))
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk/models/network_def"))
import yaml

from FaceX_Zoo.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from FaceX_Zoo.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

# 收集指定目录下的视频路径并写入CSV文件
def collect_videos_to_csv(root_dir, output_csv, video_extensions=('.mp4', '.avi', '.mov', '.mkv'), audio_label='0', visual_label='0', overall_label='0'):
    video_candidates = []

    # 先收集所有候选视频路径
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_candidates.append(os.path.join(dirpath, filename))

    print("共有 {} 个视频".format(len(video_candidates)))

    # 写入 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'audio_label', 'visual_label', 'overall_label'])  # 写入表头
        for path in video_candidates:
            writer.writerow([path, audio_label, visual_label, overall_label])

    print(f"\n共有 {len(video_candidates)} 个的视频，结果已保存到 {output_csv}")
    return output_csv


# 将单个视频裁剪为3.2s（如果超过了3.2s的话）
def split_video(input_file, segment_length=3.2):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    video = VideoFileClip(input_file)
    duration = video.duration
    num_segments = math.ceil(duration / segment_length)
    print(f"video duration:{duration}")
    start_time = 0
    if duration >= start_time+3.2:
        
        end_time = start_time + segment_length
        
        segment = video.subclip(start_time, end_time)

        output_file = f"{base_name}_split.mp4"
        segment.write_videofile(output_file, codec="libx264", audio_codec="aac")
        split = True

    else:
        output_file = f"{base_name}.mp4"
        split = False
    
    return split, output_file

# 将csv文件中的视频拆分为指定时长的片段，并写入新的CSV文件
def split_videos_from_csv(input_csv, output_csv, output_dir, segment_length=3.20):
    segment_length = float(segment_length)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 CSV
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 跳过表头
        video_data = []
        for row in reader:
            video_data.append(row)

    # 写入 CSV 表头
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_path", "audio_label", "visual_label", "overall_label"])

    # 处理每个视频
    with tqdm(total=len(video_data), desc="Processing Videos", unit="video") as pbar:
        for video_path, audio_label, visual_label, overall_label in video_data:
            if not os.path.exists(video_path):
                print(f"file {video_path} do not exist，pass...")
                pbar.update(1)
                continue

            # 生成唯一名称
            relative_path = os.path.relpath(video_path, start=os.path.commonpath([video_path, output_dir]))
            safe_name = relative_path.replace('/', '_').replace('\\', '_')
            base_name = os.path.splitext(safe_name)[0]

            try:
                video = VideoFileClip(video_path)
                duration = video.duration
            except Exception as e:
                print(f"无法读取视频 {video_path}，错误: {e}")
                pbar.update(1)
                continue

            if duration <= segment_length:
                continue
            else:
                # 拆分成多段
                num_segments = int(duration // segment_length)
                for i in range(num_segments):
                    start_time = i * segment_length
                    end_time = min(start_time + segment_length, duration)

                    output_file = os.path.join(output_dir, f"{base_name}_part_{i + 1}.mp4")
                    if not os.path.exists(output_file):
                        try:
                            segment = video.subclip(start_time, end_time)
                            segment.write_videofile(output_file, codec="libx264", audio_codec="aac", logger=None)
                        except Exception as e:
                            print(f"写入片段失败: {output_file}，错误: {e}")
                            continue
                    # write into csv, don't forget to delete the previous csv when running this function again
                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([output_file, audio_label, visual_label, overall_label])
            video.close()
            pbar.update(1)

    print(f"视频分割完成，结果已保存至 {output_csv}")

# 视频5Hz采样函数（通过对3.2s的视频均匀采样16帧实现）
def sample_video_uniform_16_from_csv_decord(input_csv, output_csv, frame_output_root='./sampled_frames'):
    os.makedirs(frame_output_root, exist_ok=True)

    # 初始化CSV文件（只包含视频文件夹路径和标签）
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['face_crop_folder', 'audio_label', 'visual_label', 'overall_label'])

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        #reader = csv.DictReader(csvfile) # 由于 csv.DictReader 是惰性迭代器，它本身无法预先知道总行数
        reader = list(csv.DictReader(csvfile))
        for row in tqdm(reader, desc="使用Decord均匀采样视频", total=len(reader)):
            video_path = row['video_path']
            audio_label = row.get('audio_label', '0')
            visual_label = row.get('visual_label', '0')
            overall_label = row.get('overall_label', '0')
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join(frame_output_root, video_name)
            
            if os.path.exists(frame_dir):
                existing_frames = [f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg')]
                if len(existing_frames) >= 16:
                    print(f"已存在帧目录且帧数 >=16，跳过: {frame_dir}")
                    continue
            else:
                os.makedirs(frame_dir, exist_ok=True)

            try:
                vr = VideoReader(video_path)
            except Exception as e:
                print(f"无法打开视频: {video_path}，错误: {e}")
                continue

            total_frames = len(vr)
            if total_frames < 16:
                print(f"视频帧数不足16帧，跳过: {video_path}")
                continue

            sample_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
            frames = vr.get_batch(sample_indices).asnumpy()

            for idx, frame in enumerate(frames):
                frame_path = os.path.join(frame_dir, f"frame_{idx:02d}.png")
                imageio.imwrite(frame_path, frame)

            # 每处理一个视频就写入一次：视频帧文件夹路径和标签
            with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([frame_dir, audio_label, visual_label, overall_label])

    print(f"采样完成，结果已逐步写入到 {output_csv}")

# 对csv文件中的视频进行5Hz采样
def sample_video_uniform_16_decord(video_path, frame_output_root='./sampled_frames'):
    """
    Args:
        video_path (str): 输入视频路径
        frame_output_root (str): 保存帧图像的根目录
    """
    os.makedirs(frame_output_root, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = frame_output_root

    if os.path.exists(frame_dir):
        existing_frames = [f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(existing_frames) >= 16:
            print(f"已存在帧目录且帧数 >=16，跳过: {frame_dir}")
            return
    else:
        os.makedirs(frame_dir, exist_ok=True)

    try:
        vr = VideoReader(video_path)
    except Exception as e:
        print(f"无法打开视频: {video_path}，错误: {e}")
        return

    total_frames = len(vr)
    if total_frames < 16:
        print(f"视频帧数不足16帧，跳过: {video_path}")
        return

    sample_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    frames = vr.get_batch(sample_indices).asnumpy()

    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frame_dir, f"frame_{idx:02d}.png")
        imageio.imwrite(frame_path, frame)

    print(f"帧采样完成: {video_path} → {frame_dir}")

# 加载FaceX-Zoo人脸检测模型
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
    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    
    if faceDetModelHandler is not None and faceDetModelLoader is not None:
        print("Face detection model loaded successfully.")
    return faceDetModelHandler

# 类: 人脸区域提取器
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

    def process_csv(self, input_csv, output_csv, save_root='./face_crops'):
        """
        批量处理视频帧文件夹，对其中前16帧提取人脸图像保存。
        每处理完一个视频就记录其人脸文件夹路径及标签到 CSV。
        """
        os.makedirs(save_root, exist_ok=True)

        # 初始化输出 CSV 文件（带表头）
        if not os.path.exists(output_csv):
            with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(['face_crop_folder', 'audio_label', 'visual_label', 'overall_label'])


        # 逐行读取输入 CSV
        with open(input_csv, mode='r', encoding='utf-8') as f_in:
            reader = list(csv.DictReader(f_in))
            for row in tqdm(reader, desc="批量提取人脸区域", total=len(reader)):
                frame_folder = row['video_folder']
                audio_label = row.get('audio_label', '0')
                visual_label = row.get('visual_label', '0')
                overall_label = row.get('overall_label', '0')
                video_name = os.path.basename(frame_folder.rstrip('/\\'))
                target_folder = os.path.join(save_root, video_name)

                # 如果已存在且图像充足，跳过
                if os.path.exists(target_folder):
                    existing = [f for f in os.listdir(target_folder) if f.endswith(('.jpg', '.png'))]
                    if len(existing) >= 16:
                        print(f"已存在且图像足够，跳过: {target_folder}")
                        continue

                os.makedirs(target_folder, exist_ok=True)

                try:
                    frame_paths = sorted([
                        os.path.join(frame_folder, f)
                        for f in os.listdir(frame_folder)
                        if f.lower().endswith(('.jpg', '.png'))
                    ])
                    if len(frame_paths) < 16:
                        print(f"跳过：{frame_folder} 中帧数不足 16")
                        continue

                    for i, frame_path in enumerate(frame_paths[:16]):
                        save_path = os.path.join(target_folder, f'face_{i:02d}.png')
                        self.extract(frame_path, save_path)

                    # 写入 CSV：已成功处理该视频
                    with open(output_csv, mode='a', newline='', encoding='utf-8') as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow([target_folder, audio_label, visual_label, overall_label])

                except Exception as e:
                    print(f" 处理失败 {frame_folder}: {e}")
                    continue

        print(f" 人脸批处理完成，结果已写入: {output_csv}")

def create_face_cropper(scale=1):
    face_detection_model = load_face_detection_model()
    face_cropper = FaceX_Zoo_FaceCropper(faceDetModelHandler=face_detection_model, scale=scale)
    return face_cropper

# 生成匹配的CSV文件，包含视频路径(用于提取音频)、人脸文件夹路径及标签
def generate_matched_csv(video_csv, face_csv, output_csv):
    # 读取视频路径和标签
    video_info = {}
    with open(video_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row['video_path']
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            video_info[video_basename] = {
                'video_path': video_path,
                'audio_label': row['audio_label'],
                'visual_label': row['visual_label'],
                'overall_label': row['overall_label'],
            }

    # 读取人脸路径并尝试匹配
    matched_rows = []
    with open(face_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            face_folder = row['face_crop_folder']
            face_basename = os.path.basename(face_folder)
            if face_basename in video_info:
                matched_rows.append({
                    'video_path': video_info[face_basename]['video_path'],
                    'face_crop_folder': face_folder,
                    'audio_label': video_info[face_basename]['audio_label'],
                    'visual_label': video_info[face_basename]['visual_label'],
                    'overall_label': video_info[face_basename]['overall_label']
                })
            else:
                print(f"[跳过] 无法匹配：{face_basename}")

    # 写入新 CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_path', 'face_crop_folder', 'audio_label', 'visual_label', 'overall_label'])
        writer.writeheader()
        writer.writerows(matched_rows)
    print(f"[完成] 匹配结果保存到 {output_csv}，共 {len(matched_rows)} 条")

# 视频处理流水线函数
def pipeline_process(csv_path):
    base_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    prefix = base_name.split('.')[0]

    split_csv = os.path.join(base_dir, f"{prefix}_split.csv")
    split_save_root = os.path.join(base_dir, f"{prefix}_split")

    print(f"Split videos: {csv_path} -> {split_csv}")
    split_videos_from_csv(csv_path, split_csv, split_save_root)

    sampled_csv = split_csv.replace("_split.csv", "_split_sampled.csv")
    sampled_save_root = split_save_root + "_sampled"

    print(f"Sample videos uniformly (16 frames): {split_csv} -> {sampled_csv}")
    sample_video_uniform_16_from_csv_decord(split_csv, sampled_csv, sampled_save_root)
    
    face_cropper = create_face_cropper(scale=1.3)

    face_csv = sampled_csv.replace("_sampled.csv", "_sampled_face.csv")
    face_save_root = sampled_save_root + "_face"

    print(f"Face cropping: {sampled_csv} -> {face_csv}")
    face_cropper.process_csv(
        input_csv=sampled_csv,
        output_csv=face_csv,
        save_root=face_save_root
    )

    # 这里改为给文件名前加 matched_
    matched_csv = os.path.join(
        os.path.dirname(face_csv),
        "matched_" + os.path.basename(face_csv)
    )
    print(f"Generate matched CSV: {sampled_csv} + {face_csv} -> {matched_csv}")
    generate_matched_csv(split_csv, face_csv, matched_csv)

    return matched_csv