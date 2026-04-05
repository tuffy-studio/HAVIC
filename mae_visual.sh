#!/bin/bash

# Path to the pretrained model weights
PRETRAIN_PATH=

# Input video file (.mp4) for MAE visualization
INPUT_FILE=

# Directory to save the visualization results
OUTPUT_ROOT=

CUDA_VISIBLE_DEVICES=0 \
python MAE_visualization.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --input_file ${INPUT_FILE} \
    --output_root ${OUTPUT_ROOT}
