# Copyright (c) Jielun Peng, Harbin Institute of Technology.
# All rights reserved.
from video_engine import *
import os
import argparse

# 收集指定目录下的视频路径并写入CSV文件
def collect_videos_to_csv(root_dir, output_csv, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
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
        writer.writerow(['video_name', 'label'])  # 写入表头
        for path in video_candidates:
            writer.writerow([path, '0'])

    print(f"\n共有 {len(video_candidates)} 个的视频，结果已保存到 {output_csv}")
    return output_csv

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Data Engine Pipeline")
    parser.add_argument('--input', type=str, required=True, help='the root directory or csv containing videos to be processed')
    args = parser.parse_args()
    
    # 判断输入是目录还是CSV文件
    if os.path.isdir(args.input):
        print(f"Collecting videos from directory: {args.input}")
        original_csv = collect_videos_to_csv(root_dir=args.input, output_csv=os.path.join(args.input, "collected_videos.csv"))
        print(f"Video collection finished, csv saved at: {original_csv}")
    else:
        print(f"Using provided CSV file: {args.input}")
        original_csv = args.input

    matched_csv = pipeline_process(original_csv)
    print(f"Video preprocess pipeline finished, matched csv saved at: {matched_csv}")