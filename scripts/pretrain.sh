#!/bin/bash

batch_size=112
max_lr=1.5e-4
weight_decay=5e-2
beta_1=0.95
beta_2=0.999
total_epochs=200
warm_up_ratio=0.1
restart_epoch=1
restart_step=1
if_restart_train=False
saved_optimizer_path=""

num_workers=4

cl_loss_weight=0.01
rec_loss_weight=1.0
cross_loss_weight=1.0

dataset_mean=-6.9960
dataset_std=3.1205
target_length=1024
img_res=224

train_data=
validate_data=
pretrain_path=

save_dir=
save_model=True

mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -W ignore ../src/run_pretrain.py \
    --data_train "${train_data}" \
    --data_val "${validate_data}" \
    --save_dir "${save_dir}" \
    $( [ "$save_model" = "True" ] && echo "--save_model" ) \
    --max_lr "${max_lr}" \
    --total_epochs "${total_epochs}" \
    --batch_size "${batch_size}" \
    --num_workers "${num_workers}" \
    --dataset_mean "${dataset_mean}" \
    --dataset_std "${dataset_std}" \
    --target_length "${target_length}" \
    --im_res "${img_res}" \
    --pretrain_path "${pretrain_path}" \
    --cl_loss_weight "${cl_loss_weight}" \
    --rec_loss_weight "${rec_loss_weight}" \
    --cross_loss_weight "${cross_loss_weight}" \
    --restart_epoch "${restart_epoch}" \
    --restart_step "${restart_step}" \
    --saved_optimizer_path "${saved_optimizer_path}" \
    --warm_up_ratio "${warm_up_ratio}" \
    --weight_decay "${weight_decay}" \
    --beta_1 "${beta_1}" \
    --beta_2 "${beta_2}" \
    $( [ "$if_restart_train" = "True" ] && echo "--if_restart_train" ) 