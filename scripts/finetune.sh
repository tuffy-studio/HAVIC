#!/bin/bash

# ==================== Paths ====================
pretrain_path="./weights/model_to_be_ft.pth"
tr_data=   
te_data=    
save_dir=
save_model=True

mkdir -p $save_dir
mkdir -p ${save_dir}/models

lr=1e-5
head_lr=10.0
n_epochs=50
batch_size=32
num_workers=2
n_print_steps=100
weighted_sampling=False

dataset_mean=-6.9960
dataset_std=3.1205
target_length=1024
freqm=20
timem=100
audio_augment=False
visual_augment=False

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -W ignore ../src/run_finetune.py \
    --data_train ${tr_data} \
    --data_val ${te_data} \
    --save_dir ${save_dir} \
    --save_model \
    --lr ${lr} \
    --head_lr ${head_lr} \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --n_print_steps ${n_print_steps} \
    --dataset_mean ${dataset_mean} \
    --dataset_std ${dataset_std} \
    --target_length ${target_length} \
    --freqm ${freqm} \
    --timem ${timem} \
    $( [ "$weighted_sampling" = "True" ] && echo "--weighted_sampling" ) \
    $( [ "$visual_augment" = "True" ] && echo "--visual_augment" ) \
    $( [ "$audio_augment" = "True" ] && echo "--audio_augment" ) \
    $( [ "$save_model" = "True" ] && echo "--save_model" ) \
    --pretrain_path ${pretrain_path}
