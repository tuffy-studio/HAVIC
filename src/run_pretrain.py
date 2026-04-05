import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import VideoAudioDataset_Pretraining
from models.HAVIC import *
from traintest_pretrain import train

parser = argparse.ArgumentParser(description='HAVIC Pretraining Script')

# =======================
# Training Hyperparameters
# =======================
parser.add_argument('--weights_path', default=None, type=str,
                    help='Path to pretrained weights for initialization.')

parser.add_argument('--if_use_amp', action='store_true',
                    help='Enable mixed precision training.')

parser.add_argument('--total_epochs', default=200, type=int)
parser.add_argument('--warm_up_ratio', default=0.1, type=float)
parser.add_argument('--accumulation_steps', default=1, type=int)

# ======================
# Data
# ======================
parser.add_argument('--data_train', type=str, required=True,
                    help='Path to training CSV file.')
parser.add_argument('--data_val', type=str, required=True,
                    help='Path to validation CSV file.')

parser.add_argument('--target_length', default=1024, type=int)
parser.add_argument('--dataset_mean', default=-5.081, type=float)
parser.add_argument('--dataset_std', default=4.4849, type=float)
parser.add_argument('--im_res', default=224, type=int)

# ======================
# DataLoader
# ======================
parser.add_argument('--batch_size', default=112, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# ======================
# Optimizer
# ======================
parser.add_argument('--max_lr', default=1.5e-4, type=float)
parser.add_argument('--weight_decay', default=0.05, type=float)
parser.add_argument('--beta_1', default=0.95, type=float)
parser.add_argument('--beta_2', default=0.999, type=float)


# ======================
# Resume
# ======================
parser.add_argument('--if_restart_train', action='store_true')
parser.add_argument('--saved_checkpoint_path', default="", type=str)


# ======================
# Save
# ======================
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_dir', default='checkpoints', type=str)

# ======================
# Loss
# ======================
parser.add_argument('--cl_loss_weight', type=float, default=0.01)
parser.add_argument('--rec_loss_weight', type=float, default=1.0)
parser.add_argument('--cross_loss_weight', type=float, default=1.0)

parser.add_argument("--n_print_steps", default=50, type=int)

args = parser.parse_args()

# ======================
# Config
# ======================
conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': 0,
    'timem': 0,
    'mode': 'train',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'skip_norm': False,
    'im_res': args.im_res
}

val_conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': 0,
    'timem': 0,
    'mode': 'eval',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'skip_norm': False,
    'im_res': args.im_res
}

# ======================
# DataLoader
# ======================
train_loader = DataLoader(
    VideoAudioDataset_Pretraining(args.data_train, conf),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True
)

val_loader = DataLoader(
    VideoAudioDataset_Pretraining(args.data_val, val_conf),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=True
)

# ======================
# Model Init
# ======================
model = HAVIC_PT()
model_dict = model.state_dict()

# ======================
# Load weights
# ======================
if not args.if_restart_train:
    loaded_weights = torch.load(args.weights_path, map_location='cpu')
else:
    args.checkpoint = torch.load(args.saved_checkpoint_path, map_location='cpu')
    loaded_weights = args.checkpoint["model"]

# filter matched params
pretrained_dict = {
    k: v for k, v in loaded_weights.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# ======================
# Logging
# ======================
missing_keys = [k for k in model_dict.keys() if k not in loaded_weights]
unexpected_keys = [k for k in loaded_weights.keys() if k not in model_dict]

print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")
print(f"Loaded params: {len(pretrained_dict)}")

if not args.if_restart_train:
    print(f"Loaded pretrained weights from {args.weights_path}")
else:
    print(f"Resumed from {args.saved_checkpoint_path}")

# ======================
# Save dir
# ======================
os.makedirs(args.save_dir, exist_ok=True)
print(f"checkpoint save dir: {args.save_dir}")

# ======================
# Train
# ======================
print(f"Start training for {args.total_epochs} epochs")
train(model, train_loader, val_loader, args)