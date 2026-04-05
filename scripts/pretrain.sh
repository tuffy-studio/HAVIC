#!/usr/bin/env bash
set -e 

# =======================
# Training Hyperparameters
# =======================
WEIGHTS_PATH="../weights/model_to_be_pt.pth"

TOTAL_EPOCHS=200
WARMUP_RATIO=0.1

IF_USE_AMP=true # Whether to enable automatic mixed precision training
ACCUMULATION_STEPS=1 # gradient accumulation interval
 
# ======================
# Optimizer
# ======================
BATCH_SIZE=112
MAX_LR=1.5e-4
WEIGHT_DECAY=5e-2
BETA1=0.95
BETA2=0.999

# =======================
# Data & Dataloader
# =======================
NUM_WORKERS=4

TRAIN_DATA=""             # TODO
TEST_DATA=""              # TODO

DATASET_MEAN=-6.9960      # Audio feature mean (log-mel / fbank normalization)
DATASET_STD=3.1205        # Audio feature std (log-mel / fbank normalization)
TARGET_LENGTH=1024        # Audio temporal length (number of frames / time steps)

IMG_RES=224               # Video frame resolution (H = W = 224)

# =======================
# Output Configure
# =======================
SAVE_DIR=""                # TODO
IF_SAVE_MODEL=true

# =======================
# Restart / Resume
# =======================
IF_RESTART_TRAIN=false
SAVED_CHECKPOINT_PATH=""

# =======================
# Loss Weights
# =======================
CL_LOSS_WEIGHT=0.01
REC_LOSS_WEIGHT=1.0
CROSS_LOSS_WEIGHT=1.0

# =======================
# Safety Checks
# =======================
if [[ -z "${SAVE_DIR}" ]]; then
    echo "ERROR: SAVE_DIR is empty!"
    exit 1
fi

if [[ -z "${TRAIN_DATA}" || -z "${TEST_DATA}" ]]; then
    echo "ERROR: TRAIN/TEST DATA path is empty!"
    exit 1
fi

if [[ "${IF_RESTART_TRAIN}" == true && -z "${SAVED_CHECKPOINT_PATH}" ]]; then
    echo "ERROR: Restart enabled but SAVED_CHECKPOINT_PATH is empty!"
    exit 1
fi


# =======================
# Build Optional Args
# =======================
AMP_FLAG=""
[[ "${IF_USE_AMP}" == true ]] && AMP_FLAG="--if_use_amp"

SAVE_MODEL_FLAG=""
[[ "${IF_SAVE_MODEL}" == true ]] && SAVE_MODEL_FLAG="--save_model"

RESTART_FLAG=""
[[ "${IF_RESTART_TRAIN}" == true ]] && RESTART_FLAG="--if_restart_train"

# =======================
# Run Training
# =======================
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_CACHE_PATH=/dev/null \
python -W ignore ../src/run_pretrain.py \
    --weights_path "${WEIGHTS_PATH}" \
    --data_train "${TRAIN_DATA}" \
    --data_test "${TEST_DATA}" \
    --save_dir "${SAVE_DIR}" \
    ${SAVE_MODEL_FLAG} \
    ${AMP_FLAG} \
    ${RESTART_FLAG} \
    --max_lr "${MAX_LR}" \
    --total_epochs "${TOTAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --accumulation_steps "${ACCUMULATION_STEPS}"\
    --dataset_mean "${DATASET_MEAN}" \
    --dataset_std "${DATASET_STD}" \
    --target_length "${TARGET_LENGTH}" \
    --im_res "${IMG_RES}" \
    --cl_loss_weight "${CL_LOSS_WEIGHT}" \
    --rec_loss_weight "${REC_LOSS_WEIGHT}" \
    --cross_loss_weight "${CROSS_LOSS_WEIGHT}" \
    --saved_checkpoint_path "${SAVED_CHECKPOINT_PATH}" \
    --warm_up_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --beta_1 "${BETA1}" \
    --beta_2 "${BETA2}"