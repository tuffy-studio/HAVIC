import sys
import os
import csv
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

#torch.autograd.set_detect_anomaly(True)# If True, enables anomaly detection for debugging gradient explosion, but slows training

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

def get_bce_weights(labels, neg_weight=1.0, pos_weight=1.0):
    return torch.where(labels == 0, neg_weight, pos_weight).unsqueeze(-1)
        
def train(model, train_loader, test_loader, args, verbose=True): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start training model on {device}")
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time, loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    av_loss_meter, a_loss_meter, v_loss_meter = AverageMeter(), AverageMeter(), AverageMeter()

    best_epoch, best_auc, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 1, 1
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    head_params = []
    base_params = []
 
    for name, param in model.named_parameters():
            if 'audio_encoder' in name or 'visual_encoder' in name or 'AudioVisualInteractionModule' in name:
                base_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.AdamW([{'params': base_params, 'lr': args.lr, 'weight_decay': 5e-2}, 
                                  {'params': head_params, 'lr': args.lr * args.head_lr, 'weight_decay':5e-2}], 
                                   betas=(0.95, 0.999))

    base_lr = optimizer.param_groups[0]['lr']
    head_lr = optimizer.param_groups[1]['lr']
    print('base lr, head lr : ', base_lr, head_lr)

    print('Total newly initialized module parameter number is : {:.3f} million'.format(sum(p.numel() for p in head_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    

    # Cosine annealing warm restart scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,        # first cycle: 10 epochs
        T_mult=1,      # cycle length multiplier
        eta_min=1e-9,  # minimum learning rate (avoid reaching zero)
        last_epoch=-1
    )
    
    # BCE loss
    BCE_loss_fn = nn.BCEWithLogitsLoss(reduction='none') 

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...") 
    model.train()

    # logging directory
    start_time = datetime.datetime.now() + datetime.timedelta(hours=0)
    start_time_str = start_time.strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join(f"{exp_dir}/logs/", start_time_str)
    os.makedirs(log_dir, exist_ok=True)

    optimizer.zero_grad() 

    use_amp = True   # True = 使用混合精度 (autocast + GradScaler)，False = 全精度 FP32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # 如果 use_amp=False，scaler 不会起作用

    # ========================= TRAIN LOOP =========================
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, (a_input, v_input, audio_labels, video_labels, labels) in enumerate(train_loader):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            if verbose == True:
                print(f"epoch: {epoch}, train number:{i}")
                print(f"labels: {labels}")
                print(f"audio_labels: {audio_labels}")
                print(f"video_labels: {video_labels}")
            audio_labels = audio_labels.to(device)
            video_labels = video_labels.to(device)
            labels = labels.to(device)
            
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            with autocast(enabled=use_amp):
                audio_outputs, video_outputs, outputs = model(a_input, v_input)
                    
                audio_weights = get_bce_weights(audio_labels.to(device), neg_weight=1.0)
                video_weights = get_bce_weights(video_labels.to(device), neg_weight=1.0)
                weights = get_bce_weights(labels.to(device), neg_weight=1.0)
                audio_loss = (BCE_loss_fn(audio_outputs, audio_labels.unsqueeze(-1)) * audio_weights).mean()
                video_loss = (BCE_loss_fn(video_outputs, video_labels.unsqueeze(-1)) * video_weights).mean()
                av_loss = (BCE_loss_fn(outputs, labels.unsqueeze(-1)) * weights).mean()

                loss = av_loss + audio_loss + video_loss
                
                if verbose == True:
                    print(f"av_loss: {av_loss}")
                    print(f"audio_loss: {audio_loss}, video_loss: {video_loss}")
                    print(f"outputs: {torch.sigmoid(outputs).cpu().detach()}")
                    print(f"audio_outputs: {torch.sigmoid(audio_outputs).cpu().detach()}")
                    print(f"video_outputs: {torch.sigmoid(video_outputs).cpu().detach()}")

            scaler.scale(loss).backward()
            
            # === 添加梯度裁剪 ===
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            # ====================

            # 使用 GradScaler 更新参数
            scaler.step(optimizer)

            # 更新 GradScaler 的状态
            scaler.update()

            optimizer.zero_grad()
            
            loss_meter.update(loss.item(), B)
            av_loss_meter.update(av_loss.item(), B)
            a_loss_meter.update(audio_loss.item(), B)
            v_loss_meter.update(video_loss.item(), B)

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
            
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/ft_model.%d.pth" % (exp_dir, epoch))
        
        #========================================模型验证=================================
        print('start validation')
        
        stats, stats_audio, stats_video, valid_loss, valid_av_loss, valid_a_loss, valid_v_loss = \
        validate(model, test_loader)

        ap = stats[0]['ap']
        auc = stats[0]['auc']
        acc = stats[0]['acc']
        audio_ap = stats_audio[0]['ap']
        audio_auc = stats_audio[0]['auc']
        audio_acc = stats_audio[0]['acc']

        video_ap = stats_video[0]['ap']
        video_auc = stats_video[0]['auc']
        video_acc = stats_video[0]['acc']

        # 打印验证结果
        print("============================================")
        print(f"Finetuning epoch: {epoch} ")
        print("validation finished")
        print("ACC: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(auc))
        print("AP: {:.6f}".format(ap))
        print("Audio ACC: {:.6f}".format(audio_acc))
        print("Audio AUC: {:.6f}".format(audio_auc))
        print("Audio AP: {:.6f}".format(audio_ap))
        print("Video ACC: {:.6f}".format(video_acc))
        print("Video AUC: {:.6f}".format(video_auc))
        print("Video AP: {:.6f}".format(video_ap))
        print("============================================")

        # 保存验证结果到 CSV 文件
        save_data(f"{log_dir}/train_loss_ft.csv", epoch=epoch, data=loss_meter.avg, data_name="train_loss")
        save_data(f"{log_dir}/train_av_loss_ft.csv", epoch=epoch, data=av_loss_meter.avg, data_name="av_loss")
        save_data(f"{log_dir}/train_a_loss_ft.csv", epoch=epoch, data=a_loss_meter.avg, data_name="audio_loss")
        save_data(f"{log_dir}/train_v_loss_ft.csv", epoch=epoch, data=v_loss_meter.avg, data_name="video_loss")

        save_data(f"{log_dir}/test_loss_ft.csv", epoch=epoch, data=valid_loss, data_name="test_loss")
        save_data(f"{log_dir}/test_av_loss_ft.csv", epoch=epoch, data=valid_av_loss, data_name="test_av_loss")
        save_data(f"{log_dir}/test_a_loss_ft.csv", epoch=epoch, data=valid_a_loss, data_name="test_a_loss")
        save_data(f"{log_dir}/test_v_loss_ft.csv", epoch=epoch, data=valid_v_loss, data_name="test_v_loss")

        save_data(f"{log_dir}/AP.csv", epoch=epoch, data=ap, data_name="ap")
        save_data(f"{log_dir}/ACC.csv", epoch=epoch, data=acc, data_name="acc")
        save_data(f"{log_dir}/AUC.csv", epoch=epoch, data=auc, data_name="auc")

        save_data(f"{log_dir}/audio_AP.csv", epoch=epoch, data=audio_ap, data_name="audio_ap")
        save_data(f"{log_dir}/audio_ACC.csv", epoch=epoch, data=audio_acc, data_name="audio_acc")
        save_data(f"{log_dir}/audio_AUC.csv", epoch=epoch, data=audio_auc, data_name="audio_auc")
        save_data(f"{log_dir}/video_AP.csv", epoch=epoch, data=video_ap, data_name="video_ap")
        save_data(f"{log_dir}/video_ACC.csv", epoch=epoch, data=video_acc, data_name="video_acc")
        save_data(f"{log_dir}/video_AUC.csv", epoch=epoch, data=video_auc, data_name="video_auc")
        
        if auc > best_auc:
            best_epoch = epoch
            best_acc = acc
            best_auc = auc
        elif auc == best_auc:
            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc

        # 保存模型参数
        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_ft_model.pth" % (exp_dir))
            
        print('Epoch {0} learning rate: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()

        scheduler.step()

        epoch += 1

def validate(model, test_loader, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BCE_loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
    class_weights = torch.tensor([1.0, 1.0]).to(device)
    CE_loss_fn = nn.CrossEntropyLoss(weight=class_weights)#, label_smoothing=0.1)
    
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    A_audio_predictions, A_video_predictions = [], []
    A_audio_targets, A_video_targets = [], []
    av_loss_meter, a_loss_meter, v_loss_meter = [], [], []

    with torch.no_grad():
        for i, (a_input, v_input, audio_labels, video_labels, labels) in enumerate(test_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            if verbose == True:
                print(f"validate number: {i}")
                print(f"labels: {labels}")
                print(F"audio_labels: {audio_labels}")
                print(F"video_labels: {video_labels}")
            A_targets.append(labels)
            A_audio_targets.append(audio_labels)
            A_video_targets.append(video_labels)
            audio_labels = audio_labels.to(device)
            video_labels = video_labels.to(device)
            labels = labels.to(device)
 

            with autocast(enabled=False):

                audio_outputs, video_outputs, outputs = model(a_input, v_input)

                audio_weights = get_bce_weights(audio_labels.to(device), neg_weight=1.0)
                video_weights = get_bce_weights(video_labels.to(device), neg_weight=1.0)
                weights = get_bce_weights(labels.to(device), neg_weight=1.0)

                audio_loss = (BCE_loss_fn(audio_outputs, audio_labels.unsqueeze(-1)) * audio_weights).mean()
                video_loss = (BCE_loss_fn(video_outputs, video_labels.unsqueeze(-1)) * video_weights).mean()
                av_loss = (BCE_loss_fn(outputs, labels.unsqueeze(-1)) * weights).mean()


                loss = av_loss + audio_loss + video_loss

            predictions = outputs.to('cpu').detach()
            audio_predictions = audio_outputs.to('cpu').detach()
            video_predictions = video_outputs.to('cpu').detach()

            if verbose == True:
                print(f"av_loss: {av_loss}")
                print(f"audio_loss: {audio_loss}, video_loss: {video_loss}")
                print(f"outputs: {predictions}")
                print(f"audio_outputs:{audio_predictions}")
                print(f"video_outputs:{video_predictions}")

            A_predictions.append(predictions)
            A_audio_predictions.append(audio_predictions)
            A_video_predictions.append(video_predictions)

            A_loss.append(loss.to('cpu').detach())
            av_loss_meter.append(av_loss.to('cpu').detach())
            a_loss_meter.append(audio_loss.to('cpu').detach())
            v_loss_meter.append(video_loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        output = torch.cat(A_predictions)
        audio_output = torch.cat(A_audio_predictions)
        video_output = torch.cat(A_video_predictions)
        target = torch.cat(A_targets)
        audio_target = torch.cat(A_audio_targets)
        video_target = torch.cat(A_video_targets)

        loss = np.mean(A_loss)
        av_loss = np.mean(av_loss_meter)
        a_loss = np.mean(a_loss_meter)
        v_loss = np.mean(v_loss_meter)
       
        target = target.unsqueeze(1)
        audio_target = audio_target.unsqueeze(1)
        video_target = video_target.unsqueeze(1)
        stats = calculate_stats(torch.sigmoid(output).cpu(), target.cpu())
        stats_audio = calculate_stats(torch.sigmoid(audio_output).cpu(), audio_target.cpu())
        stats_video = calculate_stats(torch.sigmoid(video_output).cpu(), video_target.cpu())

        return stats, stats_audio, stats_video, loss, av_loss, a_loss, v_loss
