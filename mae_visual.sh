#!/bin/bash

# Path to the pretrained model weights
PRETRAIN_PATH="./weights/pt_model.200.pth"

# Input video file (.mp4) for MAE visualization
INPUT_FILE=""

# Directory to save the visualization results
OUTPUT_ROOT="./MAE_visualization_results"

# Masking ratios for audio and video 
audio_mask_ratio=0.8125
video_mask_ratio=0.9

CUDA_VISIBLE_DEVICES=0 \
python MAE_visualization.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --input_file ${INPUT_FILE} \
    --output_root ${OUTPUT_ROOT} \
    --audio_mask_ratio ${audio_mask_ratio} \
    --video_mask_ratio ${video_mask_ratio}
