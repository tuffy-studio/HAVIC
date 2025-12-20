#!/bin/bash

pretrain_path=
input_file=
output_root=

CUDA_VISIBLE_DEVICES=0 \
python MAE_visualization.py \
    --pretrain_path ${pretrain_path} \
    --input_file ${input_file} \
    --output_root ${output_root}

