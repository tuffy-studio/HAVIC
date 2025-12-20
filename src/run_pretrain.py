import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import VideoAudioDataset_Pretraining
from models.HAVIC import *
from traintest_pretrain import train

parser = argparse.ArgumentParser(description='HAVIC Pretraining Script')

# --- Data ---
parser.add_argument('--data-train', type=str, required=True, help='Path to the CSV file listing training videos and audio.')
parser.add_argument('--data-val', type=str, required=True, help='Path to the CSV file listing validation videos and audio.')
parser.add_argument('--target_length', default=1024, type=int, help='Target length for audio features (number of frames).')
parser.add_argument('--dataset_mean', default=-5.081, type=float, help='Mean value for audio feature normalization.')
parser.add_argument('--dataset_std', default=4.4849, type=float, help='Standard deviation for audio feature normalization.')
parser.add_argument('--im_res', default=224, type=int, help='Resolution of input images (height and width).')

# --- DataLoader ---
parser.add_argument('--batch_size', default=112, type=int, help='Number of samples per batch.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of subprocesses for data loading.')

# --- Optimizer & Scheduler ---
parser.add_argument('--max_lr', default=0.00015, type=float, help='Maximum learning rate for AdamW optimizer.')
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay for AdamW optimizer.')
parser.add_argument('--beta_1', default=0.95, type=float, help='Beta1 parameter for AdamW optimizer.')
parser.add_argument('--beta_2', default=0.999, type=float, help='Beta2 parameter for AdamW optimizer.')
parser.add_argument('--total_epochs', default=200, type=int, help='Total number of training epochs.')
parser.add_argument('--warm_up_ratio', default=0.1, type=float, help='Proportion of total steps used for learning rate warm-up.')

# --- Resume Training ---
parser.add_argument('--restart_epoch', default=1, type=int, help='Epoch number to resume training from (if restarting).')
parser.add_argument('--restart_step', default=1, type=int, help='Global step number to resume training from (if restarting).')
parser.add_argument('--if_restart_train', action='store_true', help='Restart training from a checkpoint.')
parser.add_argument('--saved_optimizer_path', default="", type=str, help='Path to a saved optimizer state (for resuming training).')

# --- Model & Checkpoint ---
parser.add_argument('--pretrain_path', default=None, type=str, help='Path to a pretrained model to initialize weights.')
parser.add_argument('--save_model', action='store_true', help='Save model checkpoints during training.')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='Directory to save model checkpoints and logs.')

# --- Loss Weights ---
parser.add_argument('--cl_loss_weight', type=float, default=0.01, help='Weight for the contrastive loss component.')
parser.add_argument('--rec_loss_weight', type=float, default=1.0, help='Weight for the reconstruction (MAE) loss component.')
parser.add_argument('--cross_loss_weight', type=float, default=1.0, help='Weight for the cross-modal loss component.')

parser.add_argument("--n_print_steps", default=50, type=int)

args = parser.parse_args()

conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mode':'train', 
            'mean':args.dataset_mean, 'std':args.dataset_std, 'im_res': args.im_res}
val_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,'mode':'eval', 
            'mean': args.dataset_mean, 'std': args.dataset_std, 'im_res': args.im_res}

# Construct dataloader
train_loader = DataLoader(VideoAudioDataset_Pretraining(args.data_train, conf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
val_loader = DataLoader(VideoAudioDataset_Pretraining(args.data_val, val_conf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

model = HAVIC_PT()
if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

# 加载checkpoint
checkpoint = torch.load(args.pretrain_path, map_location='cpu')

# 拿到当前模型的state_dict
model_dict = model.state_dict()

# 过滤checkpoint里尺寸匹配的参数
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

# 更新模型参数
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 打印哪些参数没加载上（缺失和多余的）
missing_keys = [k for k in model_dict if k not in pretrained_dict]
unexpected_keys = [k for k in checkpoint if k not in model_dict]

print(f"Missing keys (not loaded from checkpoint): {missing_keys}")
print(f"Unexpected keys (in checkpoint but not in model): {unexpected_keys}")
print(f"Loaded pretrained model from {args.pretrain_path}, loaded params: {len(pretrained_dict)}")

print("\n Creating experiment directory: %s"%args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Train model
print("Now start training for %d epochs"%args.total_epochs)
train(model, train_loader, val_loader, args)
