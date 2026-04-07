import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import *
from models.HAVIC import *
from traintest_finetune import *

parser = argparse.ArgumentParser(description='HAVIC Finetune')
parser.add_argument('--data_train', type=str, help='path to train data csv')
parser.add_argument('--data_val', type=str, help='path to val data csv')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-6.9960, type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", default=3.1205, type=float, help="the dataset audio spec std, used for input normalization")

parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--head_lr', type=float, default=10.0, help='Factor to scale learning rate for new added parts')

parser.add_argument('--n-epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--save_dir', default='checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrain_path', default=None, type=str, help='path to pretrain model')

parser.add_argument('--save_model', action='store_true', help='Whether to save model checkpoints.')
parser.add_argument('--audio_augment', action='store_true', help='Using audio augmentation')
parser.add_argument('--visual_augment', action='store_true', help='Using visual augmentation')
parser.add_argument('--weighted_sampling', action='store_true', help='Use weighted sampling')

parser.add_argument("--n_print_steps", default=100, type=int)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

args = parser.parse_args()

im_res = 224
conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode':'train', 
            'mean':args.dataset_mean, 'std':args.dataset_std, 'audio_augment':args.audio_augment, 'visual_augment':args.visual_augment, 'im_res': im_res}

val_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0,'mode':'eval', 
            'mean': args.dataset_mean, 'std': args.dataset_std, 'audio_augment':False, 'visual_augment':False, 'im_res': im_res}


# Construct dataloader
train_dataset = VideoAudioDataset_Finetuning(args.data_train, conf)
val_dataset = VideoAudioDataset_Finetuning(args.data_val, val_conf)

if args.weighted_sampling:
    print("Using weighted sampling for training dataset")
    sampler = train_dataset.get_comb_weighted_sampler()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=False)

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)

print(f"Using Train: {len(train_loader)}, Eval: {len(val_loader)}")

# Construct model
ft_model = HAVIC_FT()

# init model
if args.pretrain_path is not None:
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    miss, unexpected = ft_model.load_state_dict(mdl_weight, strict=False)
    print("Missing: ", miss)
    print("Unexpected: ", unexpected)
    print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
else:
    print("Note you are finetuning a model without any pretraining.")
    
print("\n Creating experiment directory: %s"%args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Train model
print("Now start training for %d epochs"%args.n_epochs)
train(ft_model, train_loader, val_loader, args)
