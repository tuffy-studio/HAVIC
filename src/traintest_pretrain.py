import os
import csv
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于服务器
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torch.nn.utils import clip_grad_norm_

#torch.autograd.set_detect_anomaly(True) # 若为True，则开启异常检测，追踪模型发散原因，但会影响训练速度

def save_video_frames(tensor, folder):
    """
    保存视频帧为图片。tensor 形状应为 [C, T, H, W]，值范围 0~1。
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
    绘制并保存梅尔滤波器图。

    参数：
        fbank: torch.Tensor ，形状为 [T, mel_bins]
        title: 图标题
        save_path: 保存路径
    """
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.detach().cpu().numpy()

    if fbank.ndim == 3:
        fbank = fbank.squeeze(0)  # 去掉 channel 维度

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
    # 如果 CSV 文件不存在，就创建并写入表头
    try:
        # 打开 CSV 文件，追加模式（'a'）避免覆盖原数据
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 如果文件为空，写入表头
            if file.tell() == 0:  # 检查文件是否为空
                writer.writerow(['epoch', data_name])  # 表头

            # 写入当前 epoch 和损失值
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
    
    # 时间计数器
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    # 指标计数器
    loss_all_meter, loss_a_meter, loss_v_meter, loss_c_meter, acc_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    cross_loss_v_meter, cross_loss_a_meter = AverageMeter(), AverageMeter()

    # 最佳模型保存变量
    best_epoch, best_loss = 0, np.inf

    # 恢复训练相关变量，若从头开始训练则为1
    epoch = args.restart_epoch
    global_step = args.restart_step
    
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 提取模型中所有需要梯度更新的参数
    trainables = [p for p in model.parameters() if p.requires_grad] 
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    # AdamW优化器
    optimizer = torch.optim.AdamW(trainables, lr=args.max_lr, weight_decay=args.weight_decay, betas=(args.beta_1, args.beta_2))

    # 是否从断点继续训练
    if args.if_restart_train:
        if args.saved_optimizer_path and os.path.exists(args.saved_optimizer_path):
            print(f"Loading optimizer state from {args.saved_optimizer_path}")
            checkpoint = torch.load(args.saved_optimizer_path)
            optimizer.load_state_dict(checkpoint)
        else:
            print("No valid saved optimizer state found. Starting fresh training.")
            epoch, global_step = 1, 1
    
    # 累积梯度更新，默认为1不使用
    accumulation_steps = 1

    # linear warmup + cosine decay scheduler, we use OneCycleLR here for simplicity
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,          
        total_steps=args.total_epochs*len(train_loader)//accumulation_steps,
        pct_start=args.warm_up_ratio,         
        anneal_strategy='cos',  
        div_factor=25,
        final_div_factor=1e4,     
        last_epoch = -1 if global_step==1 else global_step, 
    )

    use_amp = True   # True = 使用混合精度 (autocast + GradScaler)，False = 全精度 FP32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # 如果 use_amp=False，scaler 不会起作用

    for param_group in optimizer.param_groups:
        print("当前学习率为:", param_group['lr'])
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))

    model.train()

    # 创建日志目录，根据当前时间戳命名
    start_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    start_time_str = start_time.strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join("./logs/pretraining/", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
 
    optimizer.zero_grad() 
    # ====================================================================================
    # 训练主循环
    while epoch < args.total_epochs + 1:
        model.train()
        print('====================================================')
        print(datetime.datetime.now()) # 打印当前的日期和时间
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        begin_time = time.time()
        end_time = time.time()
        
        for i, (a_input, v_input, _,) in enumerate(train_loader): # fbank, frames, label
            # a_input.shape:[1024, 128]
            print(f"train epoch {epoch}, train number: {i}",flush=True)
   
            assert a_input.shape[0] == v_input.shape[0] 
            B = a_input.shape[0]

            # 加载数据
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            
            data_time.update(time.time() - end_time) # batch加载时间
            per_sample_data_time.update((time.time() - end_time) / B) # 每个样本的平均加载时间

            dnn_start_time = time.time()
       
            # 自动混合精度降低显存使用
            with autocast(enabled=use_amp): 
                rec_loss_v, rec_loss_a, nce_loss, c_acc, \
                cross_emb_loss_video, cross_emb_loss_audio, \
                ids_keep_video, video_recon, \
                ids_keep_audio, audio_recon = model(a_input, v_input)

                # 计算对比损失和准确率
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
            
            scaler.scale(accumulation_loss).backward() # 使用GradScaler来处理反向传播

            # 反向传播更新模型参数
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
            
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.module.parameters(), max_norm=1.0)  # 梯度裁剪，限制梯度范数为1.0
                scaler.step(optimizer) # 使用GradScaler进行梯度更新
                scaler.update()  # 更新GradScaler的状态
                scheduler.step()
                optimizer.zero_grad() # 梯度清零

                global_step += 1

            #=== 用于可视化 ===
            if i % 200 == 99:
                origin_dir = os.path.join(log_dir, "origin_frames")
                recon_dir = os.path.join(log_dir, "reconstructed_frames")
                os.makedirs(origin_dir, exist_ok=True)
                os.makedirs(recon_dir, exist_ok=True)

                # 绘制音频梅尔频谱图（detach 后安全）
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
                # 保存视频帧
                save_video_frames(v_input[0].detach(), folder=origin_dir)
                save_video_frames(video_recon[0].detach(), folder=recon_dir)


            if global_step % 100 == 0:
                print(f"Step {global_step}: learning rate = {scheduler.get_last_lr()[0]}")
            
            # 更新指标计数器
            loss_all_meter.update(loss.item(), B)
            loss_a_meter.update(rec_loss_a.item(), B)
            loss_v_meter.update(rec_loss_v.item(), B)
            loss_c_meter.update(nce_loss.item(), B)
            acc_c_meter.update(c_acc.item(), B)
            cross_loss_a_meter.update(cross_emb_loss_audio.item(), B)
            cross_loss_v_meter.update(cross_emb_loss_video.item(), B)

            # 更新时间计数器
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = i % args.n_print_steps == 0

            if print_step and i != 0:
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

            end_time = time.time()
            
        # 记录指标
        save_data(f"{log_dir}/train_loss_audio.csv", epoch=epoch, data=loss_a_meter.avg, data_name="train_loss_audio")
        save_data(f"{log_dir}/train_loss_video.csv", epoch=epoch, data=loss_v_meter.avg, data_name="train_loss_video")
        save_data(f"{log_dir}/train_contrastive_loss.csv", epoch=epoch, data=loss_c_meter.avg, data_name="train_contrastive_loss")
        save_data(f"{log_dir}/train_contrastive_acc.csv", epoch=epoch, data=acc_c_meter.avg, data_name="train_contrastive_acc")
        save_data(f"{log_dir}/train_loss.csv", epoch=epoch, data=loss_all_meter.avg, data_name="train_loss")
        save_data(f"{log_dir}/train_cross_loss_audio.csv", epoch=epoch, data=cross_loss_a_meter.avg, data_name="train_cross_loss_audio")
        save_data(f"{log_dir}/train_cross_loss_video.csv", epoch=epoch, data=cross_loss_v_meter.avg, data_name="train_cross_loss_video")
        save_data(f"{log_dir}/global_step.csv", epoch=epoch, data=global_step, data_name="global_step")

        # 如果每轮都保存模型
        if args.save_model:
            torch.save(model.state_dict(), f"{exp_dir}/models/pt_model.{epoch}.pth")
            torch.save(optimizer.state_dict(), f"{exp_dir}/models/pt_model.{epoch}.optim_state.pth")
        # ====================================================================================
        #模型验证阶段
        print("Start validation...")

        # 获取验证集上的损失与指标
        eval_loss_total, eval_loss_mae_v, eval_loss_mae_a,\
        eval_loss_c, eval_c_acc,\
        eval_cross_loss_v, eval_cross_loss_a \
         = validate(model, test_loader, rec_loss_weight=rec_loss_weight, contrastive_loss_weight=contrastive_loss_weight, cross_loss_weight=cross_loss_weight)  

        # 保存验证集损失到 CSV
        save_data(f"{log_dir}/test_loss_audio.csv", epoch=epoch, data=eval_loss_mae_a, data_name="test_loss_audio")
        save_data(f"{log_dir}/test_loss_video.csv", epoch=epoch, data=eval_loss_mae_v, data_name="test_loss_video")
        save_data(f"{log_dir}/test_contrastive_loss.csv", epoch=epoch, data=eval_loss_c, data_name="test_contrastive_loss")
        save_data(f"{log_dir}/test_contrastive_acc.csv", epoch=epoch, data=eval_c_acc, data_name="test_contrastive_acc")
        save_data(f"{log_dir}/test_loss.csv", epoch=epoch, data=eval_loss_total, data_name="test_loss")
        save_data(f"{log_dir}/test_cross_loss_audio.csv", epoch=epoch, data=eval_cross_loss_a, data_name="test_cross_loss_audio")
        save_data(f"{log_dir}/test_cross_loss_video.csv", epoch=epoch, data=eval_cross_loss_v, data_name="test_cross_loss_video")

        # 打印验证集指标
        print(f"Eval Total Loss:           {eval_loss_total:.6f}")
        print(f"Eval Visual MAE Loss:      {eval_loss_mae_v:.6f}")
        print(f"Eval Audio MAE Loss:       {eval_loss_mae_a:.6f}")
        print(f"Eval Contrastive Loss:     {eval_loss_c:.6f}")
        print(f"Eval Contrastive Accuracy: {eval_c_acc:.6f}")
        print(f"Eval cross Loss Visual:    {eval_cross_loss_v:.6f}")
        print(f"Eval cross Loss Audio:     {eval_cross_loss_a:.6f}")       

        # 打印训练集指标
        print(f"Train Total Loss:           {loss_all_meter.avg:.6f}")
        print(f"Train Visual MAE Loss:      {loss_v_meter.avg:.6f}")
        print(f"Train Audio MAE Loss:       {loss_a_meter.avg:.6f}")
        print(f"Train Contrastive Loss:     {loss_c_meter.avg:.6f}")
        print(f"Train Contrastive Accuracy: {acc_c_meter.avg:.6f}")
        print(f"Train cross Loss Visual:    {cross_loss_v_meter.avg:.6f}")
        print(f"Train cross Loss Audio:     {cross_loss_a_meter.avg:.6f}")

        # 如果验证总损失更优，保存为最佳模型
        if eval_loss_total < best_loss:
            best_loss = eval_loss_total
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), f"{exp_dir}/models/best_pt_model.pth")
            torch.save(optimizer.state_dict(), f"{exp_dir}/models/best_pt_optim_state.pth")

        # 打印当前学习率
        print(f"Epoch-{epoch} lr: {optimizer.param_groups[0]['lr']:.6e}")

        # 记录本轮训练耗时
        finish_time = time.time()
        print(f"Epoch {epoch} training time: {finish_time - begin_time:.3f} seconds")

        # 每轮epoch重置各个变量
        epoch += 1

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

def validate(model, val_loader, rec_loss_weight=1, contrastive_loss_weight=0.01, cross_loss_weight=1):
    
    rec_loss_weight = rec_loss_weight
    contrastive_loss_weight = contrastive_loss_weight 
    cross_loss_weight = cross_loss_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    A_loss, A_loss_c, A_c_acc, A_loss_mae_a, A_loss_mae_v = [], [], [], [], []
    A_cross_loss_v, A_cross_loss_a = [], []

    with torch.no_grad():
        for i, (a_input, v_input, _,) in enumerate(val_loader):
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            with autocast(enabled=False):

                rec_loss_v, rec_loss_a, nce_loss, c_acc, \
                cross_emb_loss_video, cross_emb_loss_audio, \
                _, _, _, _,  \
                = model(a_input, v_input)

                # 计算对比损失和准确率
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

    # 计算所有 batch 的平均值
    loss = np.mean(A_loss)
    loss_mae_a = np.mean(A_loss_mae_a)
    loss_mae_v = np.mean(A_loss_mae_v)
    loss_c = np.mean(A_loss_c)
    c_acc = np.mean(A_c_acc)
    cross_loss_video = np.mean(A_cross_loss_v)
    cross_loss_audio = np.mean(A_cross_loss_a)    

    return loss, loss_mae_v, loss_mae_a, loss_c, c_acc, cross_loss_video, cross_loss_audio
