import os
import csv
import datetime
import time
from utilities import *
import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torch.nn.utils import clip_grad_norm_

#torch.autograd.set_detect_anomaly(True) # If True, enables anomaly detection for debugging gradient explosion, but slows training

def save_video_frames(tensor, folder):
    """
    Save video frames as images.
    Input tensor should be [C, T, H, W] with values in range [0, 1].
    """
    os.makedirs(folder, exist_ok=True)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    # tensor: [C, T, H, W] → [T, C, H, W]
    frames = tensor.permute(1, 0, 2, 3)  # (T, C, H, W)

    for i, frame in enumerate(frames):
        frame = frame.clamp(0, 1).numpy() * 255  # → (C, H, W)
        frame = frame.astype(np.uint8).transpose(1, 2, 0)  # → (H, W, C)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, f"frame_{i:02d}.png"), frame)
        
def visualize_16_frames(folder, save_path):
    frame_files = sorted(os.listdir(folder))[:16]

    if len(frame_files) < 16:
        print(f"[Warning] Not enough frames in {folder} to visualize (found {len(frame_files)})")
        return

    imgs = []
    for fname in frame_files:
        image_path = os.path.join(folder, fname)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Warning] Failed to read image: {image_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(img))

    if len(imgs) < 16:
        print(f"[Warning] Only {len(imgs)} valid images to visualize. Skipping.")
        return

    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_fbank(fbank, title="Mel Filter Bank", save_path="./mel_fbank.png"):
    """
    Plot and save mel filter bank visualization.

    Args:
        fbank (torch.Tensor): Shape [T, mel_bins]
        title (str): Plot title
        save_path (str): Output image path
    """
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.detach().cpu().numpy()

    if fbank.ndim == 3:
        fbank = fbank.squeeze(0) 

    plt.figure(figsize=(10, 4))
    plt.imshow(fbank.T, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Mel Bin Index')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_data(csv_file, epoch, data, data_name):
    """
    Save training metrics to a CSV file.

    Args:
        csv_file (str): Path to CSV file
        epoch (int): Current epoch index
        data (float): Metric value (e.g., loss, accuracy)
        data_name (str): Column name for the metric
    """
    try:
        # Open CSV file in append mode to avoid overwriting existing data
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header if file is empty
            if file.tell() == 0:
                writer.writerow(['epoch', data_name])

            # Write current epoch and value
            writer.writerow([epoch, data])

    except Exception as e:
        print(f"Error while saving data: {e}")

def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    print(f"Start pre-training model on {device}")

    rec_loss_weight=args.rec_loss_weight

    contrastive_loss_weight=args.cl_loss_weight

    cross_loss_weight = args.cross_loss_weight
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    loss_all_meter, loss_a_meter, loss_v_meter, loss_c_meter, acc_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    cross_loss_v_meter, cross_loss_a_meter = AverageMeter(), AverageMeter()

    # best model tracking variables
    best_epoch, best_loss = 0, np.inf

    epoch = 1
    accumulation_steps = args.accumulation_steps
    exp_dir = args.save_dir
    
    # wrap model for multi-GPU training
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.to(device)

    trainables = [p for p in model.parameters() if p.requires_grad] 
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    # AdamW Optimizer
    optimizer = torch.optim.AdamW(trainables, lr=args.max_lr, weight_decay=args.weight_decay, betas=(args.beta_1, args.beta_2))

    # linear warmup + cosine decay scheduler, we use OneCycleLR here for simplicity
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=args.total_epochs * len(train_loader) // accumulation_steps,
        pct_start=args.warm_up_ratio,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
    )

    use_amp = args.if_use_amp   
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # resume training
    if args.if_restart_train:
        if args.saved_checkpoint_path and os.path.exists(args.saved_checkpoint_path):
            print(f"Loading checkpoint state from {args.saved_checkpoint_path}")
            optimizer.load_state_dict(args.checkpoint["optimizer"])
            scheduler.load_state_dict(args.checkpoint["scheduler"])
            epoch = args.checkpoint['epoch']
            if use_amp and "scaler" in args.checkpoint:
                scaler.load_state_dict(args.checkpoint["scaler"])
        else:
            print("No valid saved checkpoint found. Starting fresh training.")
    

    for param_group in optimizer.param_groups:
        print("Current learning rate:", param_group['lr'])
    
    print("Current epoch=%s" % (epoch))

    # logging directory
    start_time = datetime.datetime.now() + datetime.timedelta(hours=0)
    start_time_str = start_time.strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join(f"{exp_dir}/logs/", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
 
    optimizer.zero_grad() 

    # ========================= TRAIN LOOP =========================
    while epoch < args.total_epochs + 1:
        model.train()
        print('====================================================')
        print(datetime.datetime.now()) # 打印当前的日期和时间
        print("Current epoch=%s" % (epoch))

        begin_time = time.time()
        end_time = time.time()
        
        for i, (a_input, v_input) in enumerate(train_loader): # fbank, frames
            # data loading
            assert a_input.shape[0] == v_input.shape[0] 
            B = a_input.shape[0]

            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            
            data_time.update(time.time() - end_time) # batch loading time
            per_sample_data_time.update((time.time() - end_time) / B) # average sample loading time

            dnn_start_time = time.time()
       
            # forward
            with autocast(enabled=use_amp): 
                rec_loss_v, rec_loss_a, nce_loss, c_acc, \
                cross_emb_loss_video, cross_emb_loss_audio, \
                _, video_recon, _, audio_recon = model(a_input, v_input)

                rec_loss_v = rec_loss_v.mean()
                rec_loss_a = rec_loss_a.mean()
                nce_loss = nce_loss.mean()
                c_acc = c_acc.mean()
                cross_emb_loss_video = cross_emb_loss_video.mean()
                cross_emb_loss_audio = cross_emb_loss_audio.mean()

                loss = rec_loss_weight * (rec_loss_v + rec_loss_a) \
                       + contrastive_loss_weight*nce_loss\
                       + cross_loss_weight * (cross_emb_loss_video + cross_emb_loss_audio)

                accumulation_loss = loss / accumulation_steps
            
            # backward
            scaler.scale(accumulation_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.module.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update() 
                scheduler.step()
                optimizer.zero_grad()


            # MAE visualization 
            if i % 200 == 99:
                origin_dir = os.path.join(log_dir, "origin_frames")
                recon_dir = os.path.join(log_dir, "reconstructed_frames")
                os.makedirs(origin_dir, exist_ok=True)
                os.makedirs(recon_dir, exist_ok=True)

                plot_fbank(
                    fbank=a_input[0].detach(),
                    title="Original Mel Filter Bank",
                    save_path=os.path.join(log_dir, "origin_fbank.png")
                )

                plot_fbank(
                    fbank=audio_recon[0].squeeze().T.detach(),  
                    title="Reconstructed Mel Filter Bank",
                    save_path=os.path.join(log_dir, "reconstructed_fbank.png")
                )

                save_video_frames(v_input[0].detach(), folder=origin_dir)
                save_video_frames(video_recon[0].detach(), folder=recon_dir)

            # logging
            if i % 100 == 99:
                print(f"Epoch {epoch} Step {i}: learning rate = {scheduler.get_last_lr()[0]}")
                print(
                    f"Epoch: [{epoch}][{i}/{int(len(train_loader))}]\t"
                    f"Per Sample Total Time {per_sample_time.avg:.5f}\t"
                    f"Per Sample Data Time {per_sample_data_time.avg:.5f}\t"
                    f"Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t"
                    f"Train Total Loss {loss_all_meter.val:.4f}\t"
                    f"Train MAE Loss Audio {loss_a_meter.val:.4f}\t"
                    f"Train MAE Loss Visual {loss_v_meter.val:.4f}\t"
                    f"Train Contrastive Loss {loss_c_meter.val:.4f}\t"
                    f"Train Contrastive Acc {c_acc:.3f}\t"
                    f"Train cross Loss Visual {cross_loss_v_meter.val:.4f}\t"
                    f"Train cross Loss Audio {cross_loss_a_meter.val:.4f}\t",                   
                    flush=True
                )
                if np.isnan(loss_all_meter.avg):
                    print("training diverged...")
                    return

            # undate meters
            loss_all_meter.update(loss.item(), B)
            loss_a_meter.update(rec_loss_a.item(), B)
            loss_v_meter.update(rec_loss_v.item(), B)
            loss_c_meter.update(nce_loss.item(), B)
            acc_c_meter.update(c_acc.item(), B)
            cross_loss_a_meter.update(cross_emb_loss_audio.item(), B)
            cross_loss_v_meter.update(cross_emb_loss_video.item(), B)

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            end_time = time.time()
            
        # save training log
        save_data(f"{log_dir}/train_loss_audio.csv", epoch=epoch, data=loss_a_meter.avg, data_name="train_loss_audio")
        save_data(f"{log_dir}/train_loss_video.csv", epoch=epoch, data=loss_v_meter.avg, data_name="train_loss_video")
        save_data(f"{log_dir}/train_contrastive_loss.csv", epoch=epoch, data=loss_c_meter.avg, data_name="train_contrastive_loss")
        save_data(f"{log_dir}/train_contrastive_acc.csv", epoch=epoch, data=acc_c_meter.avg, data_name="train_contrastive_acc")
        save_data(f"{log_dir}/train_loss.csv", epoch=epoch, data=loss_all_meter.avg, data_name="train_loss")
        save_data(f"{log_dir}/train_cross_loss_audio.csv", epoch=epoch, data=cross_loss_a_meter.avg, data_name="train_cross_loss_audio")
        save_data(f"{log_dir}/train_cross_loss_video.csv", epoch=epoch, data=cross_loss_v_meter.avg, data_name="train_cross_loss_video")

        # save checkpoint
        if args.if_save_model:
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            }, f"{exp_dir}/checkpoint/pretraining.{epoch}.checkpoint.pth")
        
        # ================= validation  =================
        print("Start validation...")

        eval_loss_total, eval_loss_mae_v, eval_loss_mae_a,\
        eval_loss_c, eval_c_acc,\
        eval_cross_loss_v, eval_cross_loss_a \
         = validate(model, test_loader, rec_loss_weight=rec_loss_weight, contrastive_loss_weight=contrastive_loss_weight, cross_loss_weight=cross_loss_weight)  

        # save test log
        save_data(f"{log_dir}/test_loss_audio.csv", epoch=epoch, data=eval_loss_mae_a, data_name="test_loss_audio")
        save_data(f"{log_dir}/test_loss_video.csv", epoch=epoch, data=eval_loss_mae_v, data_name="test_loss_video")
        save_data(f"{log_dir}/test_contrastive_loss.csv", epoch=epoch, data=eval_loss_c, data_name="test_contrastive_loss")
        save_data(f"{log_dir}/test_contrastive_acc.csv", epoch=epoch, data=eval_c_acc, data_name="test_contrastive_acc")
        save_data(f"{log_dir}/test_loss.csv", epoch=epoch, data=eval_loss_total, data_name="test_loss")
        save_data(f"{log_dir}/test_cross_loss_audio.csv", epoch=epoch, data=eval_cross_loss_a, data_name="test_cross_loss_audio")
        save_data(f"{log_dir}/test_cross_loss_video.csv", epoch=epoch, data=eval_cross_loss_v, data_name="test_cross_loss_video")

        print(f"Eval Total Loss:           {eval_loss_total:.6f}")
        print(f"Eval Visual MAE Loss:      {eval_loss_mae_v:.6f}")
        print(f"Eval Audio MAE Loss:       {eval_loss_mae_a:.6f}")
        print(f"Eval Contrastive Loss:     {eval_loss_c:.6f}")
        print(f"Eval Contrastive Accuracy: {eval_c_acc:.6f}")
        print(f"Eval cross Loss Visual:    {eval_cross_loss_v:.6f}")
        print(f"Eval cross Loss Audio:     {eval_cross_loss_a:.6f}")       

        print(f"Train Total Loss:           {loss_all_meter.avg:.6f}")
        print(f"Train Visual MAE Loss:      {loss_v_meter.avg:.6f}")
        print(f"Train Audio MAE Loss:       {loss_a_meter.avg:.6f}")
        print(f"Train Contrastive Loss:     {loss_c_meter.avg:.6f}")
        print(f"Train Contrastive Accuracy: {acc_c_meter.avg:.6f}")
        print(f"Train cross Loss Visual:    {cross_loss_v_meter.avg:.6f}")
        print(f"Train cross Loss Audio:     {cross_loss_a_meter.avg:.6f}")

        # save best checkpoint
        if eval_loss_total < best_loss:
            best_loss = eval_loss_total
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            }, f"{exp_dir}/checkpoint/pretraining.best.checkpoint.pth")

        # epoch time
        finish_time = time.time()
        print(f"Epoch {epoch} training time: {finish_time - begin_time:.3f} seconds")

        # reset meters
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()

        loss_all_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()
        acc_c_meter.reset()
        cross_loss_v_meter.reset()
        cross_loss_a_meter.reset()

        epoch += 1

def validate(model, val_loader, rec_loss_weight=1, contrastive_loss_weight=0.01, cross_loss_weight=1):
    
    rec_loss_weight = rec_loss_weight
    contrastive_loss_weight = contrastive_loss_weight 
    cross_loss_weight = cross_loss_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    A_loss, A_loss_c, A_c_acc, A_loss_mae_a, A_loss_mae_v = [], [], [], [], []
    A_cross_loss_v, A_cross_loss_a = [], []

    with torch.no_grad():
        for i, (a_input, v_input) in enumerate(val_loader):
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)


            rec_loss_v, rec_loss_a, nce_loss, c_acc, \
            cross_emb_loss_video, cross_emb_loss_audio, \
            _, _, _, _,  \
            = model(a_input, v_input)

            rec_loss_v = rec_loss_v.mean()
            rec_loss_a = rec_loss_a.mean()
            nce_loss = nce_loss.mean()
            c_acc = c_acc.mean()
            cross_emb_loss_video = cross_emb_loss_video.mean()
            cross_emb_loss_audio = cross_emb_loss_audio.mean()

            loss = rec_loss_weight * (rec_loss_v + rec_loss_a) \
                    + contrastive_loss_weight * nce_loss\
                    + cross_loss_weight * (cross_emb_loss_video + cross_emb_loss_audio)

            A_loss.append(loss.item())
            A_loss_mae_a.append(rec_loss_a.item())
            A_loss_mae_v.append(rec_loss_v.item())
            A_loss_c.append(nce_loss.item())
            A_c_acc.append(c_acc.item())
            A_cross_loss_a.append(cross_emb_loss_audio.item())
            A_cross_loss_v.append(cross_emb_loss_video.item())

    loss = np.mean(A_loss)
    loss_mae_a = np.mean(A_loss_mae_a)
    loss_mae_v = np.mean(A_loss_mae_v)
    loss_c = np.mean(A_loss_c)
    c_acc = np.mean(A_c_acc)
    cross_loss_video = np.mean(A_cross_loss_v)
    cross_loss_audio = np.mean(A_cross_loss_a)    

    return loss, loss_mae_v, loss_mae_a, loss_c, c_acc, cross_loss_video, cross_loss_audio
