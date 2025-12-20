# Copyright (c) 2025, Jielun Peng, Harbin Institute of Technology.
# All rights reserved.


import os
import numpy as np
import torch

import cv2
from glob import glob
from PIL import Image
import torchvision.transforms as T
import ffmpeg
import torchaudio
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于服务器
import matplotlib.pyplot as plt
from src.models.HAVIC import HAVIC_PT
from video_data_engine.video_engine import split_video, sample_video_uniform_16_decord, create_face_cropper
from einops import rearrange
import argparse


# ===================音频可视化相关函数===================

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

def plot_fbank_with_black_mask(fbank, title="Masked Mel Filter Bank", output_path = ''):
    """
    fbank: Tensor or ndarray, shape [time, mel_bins]
    掩蔽区域应为 -999，会在图中显示为黑色
    """
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.cpu().numpy()

    # 掩蔽值为 -999 的区域
    fbank_masked = np.ma.masked_where(fbank == -999, fbank)

    plt.figure(figsize=(10, 4))

    # 使用 colormap，并将 masked 区域设置为黑色
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='black')

    # 设置合理的 vmin/vmax 忽略 -999 的影响
    valid_min = fbank[fbank != -999].min()
    valid_max = fbank[fbank != -999].max()

    plt.imshow(fbank_masked.T, aspect='auto', origin='lower', interpolation='none',
               cmap=cmap, vmin=valid_min, vmax=valid_max)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Mel Bin Index')
    plt.tight_layout()
    title = os.path.join(output_path, title)
    plt.savefig(f'{title}.png')

def recover_from_mask(x_masked, ids_keep, mask_token, num_tokens=512):
    """
    使用给定的 mask_token（如 learnable 参数）来填充未保留的 patch。
    - x_masked: 掩蔽后的 token，形状 [N, L_keep, D]
    - ids_keep: 保留 token 的原始索引 [N, L_keep]
    - original_shape: 要恢复的形状 [N, T, H, W, D]
    - mask_token: [1, 1, D]，一个 learnable 的 nn.Parameter
    """
    N, L, D = x_masked.shape[0], num_tokens, x_masked.shape[-1]

    # 扩展 mask_token 到 [N, L, D]

    mask_token = mask_token.to(dtype=x_masked.dtype, device=x_masked.device)
    
    x_recover = mask_token.expand(N, L, D).clone()  # 注意：要 clone() 否则 inplace 会出错

    # 替换掉保留的位置
    for i in range(N):
        x_recover[i, ids_keep[i]] = x_masked[i]

    return x_recover

def save_mask_audio(x, ids_keep, output_path):
    x = x.squeeze(dim=0)
    x = x.reshape(64, 16, 8, 16).permute(0, 2, 1, 3)
    x = x.reshape(512, 256).unsqueeze(0)
    index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    x_masked = torch.gather(x, dim=1, index=index)  # [N, len_keep, D]
    mask_token = torch.full((1, 1, x.shape[2]), -999, dtype=x.dtype, device=x.device)
    x_restored = recover_from_mask(x_masked, ids_keep, mask_token)  # shape: [1, 512, 256]
    mel_blocks_recovered = x_restored.squeeze(0).reshape(64, 8, 16, 16)
    mel_blocks_recovered = mel_blocks_recovered.permute(0, 2, 1, 3)
    mel_spec_masked = mel_blocks_recovered.reshape(1024, 128)
    plot_fbank_with_black_mask(mel_spec_masked, title=f"Masked Mel Spectrogram", output_path=output_path)

# ===================视频可视化相关函数===================

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

def patchify_video(x, tubelet_size=2, patch_size=16):
    B, C, T, H, W = x.shape
    assert T % tubelet_size == 0 and H % patch_size == 0 and W % patch_size == 0

    # 通道放在最前面（为了 match 你的 unpatch_to_img）
    x = rearrange(x, "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)",
                  pt=tubelet_size, ph=patch_size, pw=patch_size)
    return x

def unpatch_to_img(x, tubelet_size=2, patch_size=16, n_patch_t=8, n_patch_h=14, n_patch_w=14, channels=3):
    """将 token 还原为视频 (B, C, T, H, W)"""
    x = rearrange(x, "b n (c p) -> b n p c", c=channels)
    x = rearrange(
        x,
        "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)",
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size,
        t=n_patch_t,
        h=n_patch_h,
        w=n_patch_w
    )
    return x

def apply_mask(x, ids_keep):
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))  # [N, len_keep, D]
    return x_masked

def save_video_frames(tensor, folder):
    frames = tensor[0].permute(1, 0, 2, 3)  # (T, C, H, W)
    frames = (frames * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder, f"frame_{i:02d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def save_mask_video(video, ids_keep_expanded, save_dir_masked):
    x_patch = patchify_video(video)
    x_masked = apply_mask(x_patch, ids_keep_expanded)
    x_recovered = recover_from_mask(
    x_masked, ids_keep_expanded, (1, 8, 14, 14, x_patch.shape[2])
    )
    video_masked = unpatch_to_img(
    x_recovered,
    tubelet_size=2, patch_size=16,
    n_patch_t=8, n_patch_h=14, n_patch_w=14, channels=3
    )  # (1, 3, 16, 224, 224)
    save_video_frames(video_masked, save_dir_masked)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Visualization')
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_root', type=str)
    args = parser.parse_args()
    
    # ====================== step1: 输入数据预处理 ======================
    input_file = args.input_file
    output_root = args.output_root
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    log_dir = os.path.join(output_root, file_name)
    os.makedirs(log_dir, exist_ok=True)
    print("log_dir:", log_dir)

    # 将视频裁剪为3.2s
    split, output_file = split_video(input_file = input_file)

    if not split:
        print("The input file is too short!!!")
        output_file = input_file
    
    sample_frame_output_root = os.path.join(log_dir, 'sample_frames')
    os.makedirs(sample_frame_output_root, exist_ok=True)
    sample_video_uniform_16_decord(video_path=output_file, frame_output_root=sample_frame_output_root)
    
    face_cropper = create_face_cropper()

    image_paths = sorted(glob(os.path.join(sample_frame_output_root, '*.png')) + glob(os.path.join(sample_frame_output_root, '*.jpg')))

    sample_face_output_root = os.path.join(log_dir, 'sample_faces')
    os.makedirs(sample_face_output_root, exist_ok=True)

    # 提取人脸
    for frame_path in image_paths:
        frame_name = os.path.basename(frame_path)
        save_path = os.path.join(sample_face_output_root, frame_name)
        try:
            face_cropper.extract(frame_path, save_path)
        except Exception as e:
            print(f"处理失败: {frame_path}，错误原因: {e}")

    # 视频处理
    frames = []

    preprocess = T.Compose([T.Resize(size=(224, 224)), T.ToTensor()])

    for i in range(16):  # num_frames = 16
        frame_path = os.path.join(sample_face_output_root, f"frame_{i:02d}.png")
        try:
            frame = Image.open(frame_path).convert('RGB')  
            frame = preprocess(frame)
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
            frame = torch.zeros(3, 224, 224)
        frames.append(frame)
    frames = torch.stack(frames)  # 输出形状: [16, 3, H, W]
    v_input = frames.permute(1, 0, 2, 3).unsqueeze(0) #[1, 3, 16, 224, 224]

    # 音频处理
    temp_audio_file = output_file.replace('.mp4', '.wav')
    ffmpeg.input(output_file).output(temp_audio_file, ac=1, ar='16k').run(quiet=True)
    waveform, sr = torchaudio.load(temp_audio_file)
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) # [T, 128]

    fbank = torch.nn.functional.interpolate(
        fbank.unsqueeze(0).transpose(1, 2), size=(1024,),
        mode='linear', align_corners=False).transpose(1, 2).squeeze(0) # [1024, 128]

    dataset_mean=-6.9960
    dataset_std=3.1205
    fbank = (fbank - dataset_mean) / (dataset_std)

    a_input = fbank.unsqueeze(0) # [1, 1024, 128]

    # ===================== step2: 加载预训练模型 =======================
    # 先构造模型并DataParallel包装（如果用的DataParallel）
    model = HAVIC_PT()
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # 加载checkpoint路径
    pretrain_path = args.pretrain_path

    # 安全检查
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    missing_keys = [k for k in model_dict if k not in pretrained_dict]
    unexpected_keys = [k for k in checkpoint if k not in model_dict]
    print(f"Missing keys (not loaded from checkpoint): {missing_keys}")
    print(f"Unexpected keys (in checkpoint but not in model): {unexpected_keys}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # ======================= step3: 模型推理 ==========================
    a_input = a_input.to(device, non_blocking=True)
    v_input = v_input.to(device, non_blocking=True)
    
    _, _, _, _, _, _, ids_keep_video, video_recon, ids_keep_audio, audio_recon = model(a_input, v_input)

    # ==================== step4: 保存并展示结果 =========================
    # 定义保存路径
    origin_dir = os.path.join(log_dir, "origin_frames")
    mask_dir = os.path.join(log_dir, "mask_frames")
    recon_dir = os.path.join(log_dir, "reconstructed_frames")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    # 原始梅尔谱图
    plot_fbank(
        fbank=a_input[0].detach(),
        title="Original Mel Spectrogram",
        save_path=os.path.join(log_dir, "origin_fbank.png")
    )

    # 掩蔽后的梅尔谱图
    save_mask_audio(a_input, ids_keep_audio, output_path=log_dir)
    

    # 重建后的梅尔谱图
    plot_fbank(
        fbank=audio_recon[0].squeeze().T.detach(),  
        title="Reconstructed Mel Spectrogram",
        save_path=os.path.join(log_dir, "reconstructed_fbank.png")
    )

    # 原始视频帧
    save_video_frames(v_input[0].detach(), folder=origin_dir)
    visualize_16_frames(folder=origin_dir, save_path=os.path.join(log_dir, "origin_frames.png"))

    # 掩蔽后的视频帧
    save_mask_video(v_input, ids_keep_video, mask_dir)
    visualize_16_frames(folder=mask_dir, save_path=os.path.join(log_dir, "mask_frames.png"))

    # 重建后的视频帧
    save_video_frames(video_recon[0].detach(), folder=recon_dir)
    visualize_16_frames(folder=recon_dir, save_path=os.path.join(log_dir, "recon_frames.png"))
