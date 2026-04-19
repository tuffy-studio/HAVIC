# Copyright (c) Jielun Peng, Harbin Institute of Technology.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .visual_modules import VisualEncoder, VisualDecoder
from .audio_modules import AudioEncoder, AudioDecoder
from .interaction_modules import AudioVisualInteractionModule, A2V_Decoder, V2A_Decoder
from .classification_modules import TokenWise_TokenReducer

class HAVIC_PT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 n_frames=16,
                 audio_length=1024,
                 mel_bins=128,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_embed_dim=384,
                 decoder_depth=4,
                 decoder_num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer="LayerNorm",
                 init_values=0.,
                 tubelet_size=2,
                 audio_mask_ratio=0.8125,
                 video_mask_ratio=0.9,
                 ):
        super().__init__()

        self.audio_mask_ratio = audio_mask_ratio
        self.video_mask_ratio = video_mask_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.patch_size = patch_size
        self.n_frames = n_frames

        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.audio_decoder = AudioDecoder(
            num_patches=audio_length * mel_bins // (patch_size ** 2),
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.visual_encoder = VisualEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.visual_decoder = VisualDecoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.AudioVisualInteractionModule = AudioVisualInteractionModule(num_layers=1)

        self.A2V = A2V_Decoder()
        self.V2A = V2A_Decoder()

    def generate_tube_mask_indices(self, N, T=8, H=14, W=14):
        mask_ratio = self.video_mask_ratio
        device = self.device
        patches_per_frame = H * W
        L = T * patches_per_frame
        len_keep_per_frame = int(patches_per_frame * (1 - mask_ratio))

        # 每帧生成随机 noise 并打乱索引
        noise = torch.rand(N, patches_per_frame, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep_one = ids_shuffle[:, :len_keep_per_frame]

        # 将每帧的索引扩展到整个序列
        ids_keep = torch.cat([
            ids_keep_one + i * patches_per_frame for i in range(T)
        ], dim=1)  # [N, len_keep]

        return ids_keep

    def generate_audio_mask_indices(self, N, L=512):
        device = self.device
        segment_len = L // 8
        mask_ratio = self.audio_mask_ratio
        len_keep_per_segment = int(segment_len * (1 - mask_ratio))

        ids_keep_all = []

        for _ in range(N):
            keep_indices = []
            for i in range(8):  # 对每个时间段进行掩蔽
                start = i * segment_len

                # 生成当前时间段内的随机噪声排序
                noise = torch.rand(segment_len, device=device)
                ids_shuffle = torch.argsort(noise)
                ids_keep_segment = ids_shuffle[:len_keep_per_segment] + start  # 加上偏移量
                keep_indices.append(ids_keep_segment)

            # 拼接每个段的保留索引
            keep_indices = torch.cat(keep_indices, dim=0)  # [len_keep]
            ids_keep_all.append(keep_indices)

        ids_keep = torch.stack(ids_keep_all, dim=0)  # [N, len_keep]

        return ids_keep

    def split_cls_patch_tokens(self, emb, n_segments=8):
        B, L, D = emb.shape
        seg_len = L // n_segments
        emb = emb.view(B, n_segments, seg_len, D)  # [B, 8, 1+P, D]

        cls_tokens = emb[:, :, 0, :]      # [B, 8, D]
        patch_tokens = emb[:, :, 1:, :]    # [B, 8, P, D]

        return cls_tokens, patch_tokens.reshape(B, -1, D)
    

    def forward_mse_loss_audio(self, audio_input, audio_recon, ids_keep, p=16, all=False):
        """
        audio_input:  [B, 1, 128, 1024]  # 原始音频频谱图，已是统一格式
        audio_recon:  [B, 1, 128, 1024]  # 解码器重建输出，unpatchify 后的结果
        ids_keep:     [B, num_keep]      # 每个样本中未被掩蔽的 patch 索引

        return:       float，MSE loss 只在掩蔽区域上计算
        """
        # patchify 原始音频和重建音频
        audio_patches      = self.patchify_audio(audio_input, p=p)      # [B, 512, 256]
        audio_recon_patches = self.patchify_audio(audio_recon, p=p)     # [B, 512, 256]

        B, N, D = audio_patches.shape
        device = audio_input.device

        loss = 0.0
        for i in range(B):
            mask = torch.ones(N, dtype=torch.bool, device=device)
            if all:
                masked_audio = audio_patches[i]
                masked_recon = audio_recon_patches[i]
            else:
                mask[ids_keep[i]] = False  # False 表示visible，True 表示masked
                # 只在masked区域计算MSE
                masked_audio = audio_patches[i][mask]
                masked_recon = audio_recon_patches[i][mask]

            loss_i = ((masked_audio - masked_recon) ** 2).mean()
            loss += loss_i

        return loss / B

    def forward_mse_loss_video(self, video, video_recon, ids_keep, all=False):
        """
        video:       [B, 3, 16, 224, 224]
        video_recon: [B, 3, 16, 224, 224]
        ids_keep:    [B, num_keep_patches],

        计算 MSE loss 仅在 **被mask** 的 patch 上。
        """

        B = video.shape[0]
        device = video.device

        # patchify，变成 [B, N, patch_dim]
        video_patches = self.patchify_video(video)         # [B, N, P]
        recon_patches = self.patchify_video(video_recon)   # [B, N, P]
        N = video_patches.shape[1]

        loss = 0.0

        for i in range(B):
            # ids_keep[i] 是已保留的patch索引，转换成布尔掩码
            mask = torch.ones(N, dtype=torch.bool, device=device)
            if all:
                masked_video = video_patches[i]
                masked_recon = recon_patches[i]
            else:
                mask[ids_keep[i]] = False  # False 表示visible，True 表示masked
                # 只在masked区域计算MSE
                masked_video = video_patches[i][mask]
                masked_recon = recon_patches[i][mask]

            loss_i = ((masked_video - masked_recon) ** 2).mean()
            loss += loss_i

        return loss / B

    def forward_contrastive(self, audio_rep, video_rep, temperature=0.07, n_frames=8, direction='bidirectional'):
        # L2 Normalize
        audio_rep = F.normalize(audio_rep, dim=-1)
        video_rep = F.normalize(video_rep, dim=-1)

        B_T, D = audio_rep.shape
        B = B_T // n_frames
        device = audio_rep.device

        # Similarity matrices
        sim_a2v = torch.matmul(audio_rep, video_rep.T) / temperature
        sim_v2a = sim_a2v.T

        # Soft temporal weights
        delta = torch.arange(n_frames, device=device).unsqueeze(0) - torch.arange(n_frames, device=device).unsqueeze(1)
        delta = delta.abs()
        w_temporal = 1 - 2 * torch.sigmoid(-delta.float())  # 你的论文公式版本
        w_temporal = w_temporal.repeat(B, B)                # [B*T, B*T]

        # Identity matrix (for self-matching)
        sample_ids = torch.arange(B, device=device).repeat_interleave(n_frames)
        same_sample = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1))  # same sample mask
        same_frame = torch.eye(B_T, dtype=torch.bool, device=device)

        # 组合最终权重矩阵
        weights = torch.ones_like(w_temporal)
        weights[same_sample & ~same_frame] = w_temporal[same_sample & ~same_frame]

        # Direction control
        sims = [sim_a2v] if direction == 'a2v' else [sim_v2a] if direction == 'v2a' else [sim_a2v, sim_v2a]

        total_loss, acc = 0, 0

        for sim in sims:
            exp_sim = torch.exp(sim)                          # numerator
            weighted_exp = exp_sim * weights                  # apply weight in denominator
            
            #log_sum_exp = torch.log(torch.sum(weighted_exp, dim=1, keepdim=True) + 1e-8)
            #prob = torch.exp(sim - log_sum_exp)  

            denom = weighted_exp.sum(dim=1, keepdim=True)     # softmax denominator
            prob = exp_sim / denom                         

            # 对角线是真正的正样本 (same sample, same frame)
            pos_mask = torch.eye(B_T, device=device)
            pos_logprob = -torch.log((prob * pos_mask).sum(dim=1) + 1e-8)

            total_loss += pos_logprob.mean()
            #total_sim += (exp_sim * pos_mask).sum(dim=1).mean()

            pred = torch.argmax(sim, dim=1)
            correct = (pred == torch.arange(B_T, device=device)).float()
            acc += correct.mean()

        return total_loss / len(sims), acc / len(sims)
    
    def patchify_video(self, imgs):
        """
        imgs: [B, 3, 16, 224, 224]
        返回:
            patches: [B, 1568, 16*16*3*2] = [B, 8*14*14, 1536]
        说明:
            - 8 是时间帧数
            - 14x14 是空间patch数
            - patch维度 = 16*16*3*2，说明每个patch由2个时间帧合并组成或双通道设计
        """

        B, C, T, H, W = imgs.shape
        p = 16
        assert H == 224 and W == 224
        assert T == 16
        H_p = H // p  # 14
        W_p = W // p  # 14

        # 先把时间维度拆成两部分，2 和 8（双帧拼接）
        # 这里的设计基于你 patch_dim = 16*16*3*2
        # 所以将时间维拆成 T=16 = 8 * 2
        imgs = imgs.view(B, C, 8, 2, H, W)  # [B, 3, 8, 2, 224, 224]

        # 然后拆空间patch，保持双帧一起拆
        imgs = imgs.view(B, C, 8, 2, H_p, p, W_p, p)  # [B,3,8,2,14,16,14,16]

        # 调整维度顺序，把patch维度排到一起，通道维放最后
        imgs = imgs.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B,8,14,14,3,2,16,16]

        # 合并patch维度：空间patch数14*14，时间“2”维作为patch内部通道扩展
        patches = imgs.reshape(B, 8 * H_p * W_p, C * 2 * p * p)  # [B, 1568, 1536]

        return patches

    def patchify_audio(self, audio, p=16):
        """
        audio: [B, 1, 128, 1024]
        输出: [B, 512, 256]，其中 512 = 8×64 是 patch 数，256 = 1×16×16 是每个 patch 展平后的特征维度
        """
        B, C, H, W = audio.shape
        assert C == 1
        assert H % p == 0 and W % p == 0, f"Input shape must be divisible by patch size {p}"

        H_p = H // p  # 8
        W_p = W // p  # 64

        # reshape 成 patch grid
        x = audio.view(B, C, H_p, p, W_p, p)  # [B, 1, 8, 16, 64, 16]

        # 调整维度，把 patch 维度挪到前面
        x = x.permute(0, 2, 4, 1, 3, 5)       # [B, 8, 64, 1, 16, 16]

        # 展平 patch 内容
        x = x.reshape(B, H_p * W_p, C * p * p)  # [B, 512, 256]

        return x

    def unpatchify(self, x, img_channel, h_token_nums, w_toke_nums, p=16):
        """
        作用：在梅尔谱图(B, 1, 128, 1024)被分割成小块处理后(B, 512, 256)，再恢复为梅尔谱图
        x: (batch_size, token_nums, channel)
        imgs: (batch_size, img_channel, h, w)
        """
        assert h_token_nums * w_toke_nums == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h_token_nums, w_toke_nums, p, p, img_channel))

        x = torch.einsum('nhwpqc->nchpwq', x)

        imgs = x.reshape(shape=(x.shape[0], img_channel, h_token_nums * p, w_toke_nums * p))
        return imgs

    def safe_mse_loss(self, x, y, reduction='mean', clamp_value=1e4):
        # 转成 float32 以防半精度溢出
        x = x.float()
        y = y.float()
        # 替换 NaN/Inf 并限幅
        x = torch.nan_to_num(x, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        y = torch.nan_to_num(y, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        x = torch.clamp(x, -clamp_value, clamp_value)
        y = torch.clamp(y, -clamp_value, clamp_value)
        # 计算 MSE
        loss = F.mse_loss(x, y, reduction=reduction)
        # 防止 loss 自身为 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in MSE loss, returning 0.")
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        return loss

    def forward(self, audio, video):

        # ======================== step1: 音频、视频编码 ========================
        ids_keep_video = self.generate_tube_mask_indices(N=video.shape[0])
        ids_keep_audio = self.generate_audio_mask_indices(N=audio.shape[0])

        video_3, video_6, video_9, video_emb = self.visual_encoder(video, ids_keep_video) # (B, 1568, 768)  
        audio_3, audio_6, audio_9, audio_emb = self.audio_encoder(audio, ids_keep_audio) # (B, 512, 768)

        video_cls_tokens, video_emb = self.split_cls_patch_tokens(video_emb, n_segments=8) # (B, 8, 768), (B, 8*14*14, 768)
        audio_cls_tokens, audio_emb = self.split_cls_patch_tokens(audio_emb, n_segments=8) # (B, 8, 768), (B, 8*64, 768)

        video_cls_tokens = video_cls_tokens.unsqueeze(dim=2)
        audio_cls_tokens = audio_cls_tokens.unsqueeze(dim=2)

        # ========================= step2: 对比学习损失 =========================
        B,_,D = video_emb.shape
        video_emb_reduced = video_emb.reshape(B,8,-1,D).reshape(B*8,-1,D).mean(dim=1)
        audio_emb_reduced = audio_emb.reshape(B,8,-1,D).reshape(B*8,-1,D).mean(dim=1)
        nce_loss, c_acc = self.forward_contrastive(audio_emb_reduced, video_emb_reduced)

        # ========================== step3: 音频、视频交互 ===========================
        video_inter, audio_inter = self.AudioVisualInteractionModule(video = video_emb, audio = audio_emb,\
                                                             video_cls_tokens = video_cls_tokens,\
                                                             audio_cls_tokens = audio_cls_tokens)

        _, video_inter = self.split_cls_patch_tokens(video_inter, n_segments=8)
        _, audio_inter = self.split_cls_patch_tokens(audio_inter, n_segments=8)                                              

        # ======================== step4: 音频、视频跨模态语义重建 ========================
        trans_video_emb = self.A2V(audio_inter, ids_keep_audio = ids_keep_audio)
        trans_audio_emb = self.V2A(video_inter, ids_keep_video = ids_keep_video)

        with torch.no_grad():
            _, _, _, full_video_emb = self.visual_encoder(video)
            _, _, _, full_audio_emb = self.audio_encoder(audio)
            
            full_audio_cls_tokens, full_audio_emb = self.split_cls_patch_tokens(full_audio_emb, n_segments=8)
            full_video_cls_tokens, full_video_emb = self.split_cls_patch_tokens(full_video_emb, n_segments=8)

        trans_emb_loss_audio = self.safe_mse_loss(trans_audio_emb, full_audio_emb)
        trans_emb_loss_video = self.safe_mse_loss(trans_video_emb, full_video_emb)

        # ======================== step5: 音频、视频重建 ========================
        _, video_3 = self.split_cls_patch_tokens(video_3, n_segments=8) # (B, 8*14*14, 768)
        _, video_6 = self.split_cls_patch_tokens(video_6, n_segments=8)
        _, video_9 = self.split_cls_patch_tokens(video_9, n_segments=8)

        _, audio_3 = self.split_cls_patch_tokens(audio_3, n_segments=8) # (B, 8*64, 768)
        _, audio_6 = self.split_cls_patch_tokens(audio_6, n_segments=8)
        _, audio_9 = self.split_cls_patch_tokens(audio_9, n_segments=8)
        
        # 视频重建
        v1,v2,v3,video_recon = self.visual_decoder(video_inter, video_3, video_6, video_9, video_emb, ids_keep_video = ids_keep_video) # (B, 1568, 768) --> (B, 1568, 16*16*3*2)

        video_recon = self.visual_decoder.unpatch_to_img(video_recon) # (B, 1568, 16*16*3*2) --> (B, 3, 16, 224, 224)
        v1 = self.visual_decoder.unpatch_to_img(v1)
        v2 = self.visual_decoder.unpatch_to_img(v2)
        v3 = self.visual_decoder.unpatch_to_img(v3)


        # 音频重建
        a1,a2,a3,audio_recon = self.audio_decoder(audio_inter, audio_3, audio_6, audio_9, audio_emb, ids_keep_audio = ids_keep_audio) # (B, 512, 768) --> (B, 512, 16*16*1)

        audio_input = audio.unsqueeze(1) #(B, 1, 1024, 128)
        audio_input = audio_input.transpose(2, 3) # (B, 1, 128, 1024)

        audio_recon = self.unpatchify(audio_recon, img_channel=1, h_token_nums=audio_input.shape[2] // 16, w_toke_nums=audio_input.shape[3] // 16, p=16) # (B, 512, 16*16*1) --> (B, 1, 128, 1024)
        a1 = self.unpatchify(a1, img_channel=1, h_token_nums=audio_input.shape[2] // 16, w_toke_nums=audio_input.shape[3] // 16, p=16)
        a2 = self.unpatchify(a2, img_channel=1, h_token_nums=audio_input.shape[2] // 16, w_toke_nums=audio_input.shape[3] // 16, p=16)
        a3 = self.unpatchify(a3, img_channel=1, h_token_nums=audio_input.shape[2] // 16, w_toke_nums=audio_input.shape[3] // 16, p=16) 

        # ======================== step5: 计算音频、视频重建损失 ========================
        rec_loss_v = self.forward_mse_loss_video(video, video_recon, ids_keep_video, all=True)  # 计算视频重建损失)
        rec_loss_v_1 = self.forward_mse_loss_video(video, v1, ids_keep_video, all=True)
        rec_loss_v_2 = self.forward_mse_loss_video(video, v2, ids_keep_video, all=True)
        rec_loss_v_3 = self.forward_mse_loss_video(video, v3, ids_keep_video, all=True)

        rec_loss_a = self.forward_mse_loss_audio(audio_input, audio_recon, ids_keep_audio, p=16, all=True) # 计算音频重建损失
        rec_loss_a_1 = self.forward_mse_loss_audio(audio_input, a1, ids_keep_audio, p=16, all=True)
        rec_loss_a_2 = self.forward_mse_loss_audio(audio_input, a2, ids_keep_audio, p=16, all=True)
        rec_loss_a_3 = self.forward_mse_loss_audio(audio_input, a3, ids_keep_audio, p=16, all=True)

        rec_loss_v = (rec_loss_v + rec_loss_v_3 + rec_loss_v_2 + rec_loss_v_1)/4
        rec_loss_a = (rec_loss_a + rec_loss_a_3 + rec_loss_a_2 + rec_loss_a_1)/4

        # Returns:
        # 1: audio reconstruction loss, video reconstruction loss, contrastive loss, contrastive accuracy
        # 2: indices of unmasked video tokens (for visualization), reconstructed video
        # 3: indices of unmasked audio tokens (for visualization), reconstructed audio
        return rec_loss_v, rec_loss_a, nce_loss, c_acc,\
               trans_emb_loss_video, trans_emb_loss_audio, \
               ids_keep_video, video_recon, \
               ids_keep_audio, audio_recon
    
class HAVIC_FT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 n_frames=16,
                 audio_length=1024,
                 mel_bins=128,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer="LayerNorm",
                 init_values=0.,
                 tubelet_size=2,
                 ):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.patch_size = patch_size
        self.n_frames = n_frames

        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.visual_encoder = VisualEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.classifier = FlexibleMLP(input_size=4 * encoder_embed_dim,
                                      hidden_sizes=[2 * encoder_embed_dim,
                                                    1 * encoder_embed_dim],
                                      num_classes=1,
                                      drop_rates=[0., 0.])
        
        self.classifier_audio = FlexibleMLP(input_size=1 * encoder_embed_dim,
                                    hidden_sizes=[2 * encoder_embed_dim,
                                                    int(1 * encoder_embed_dim)],
                                    num_classes=1,
                                    drop_rates=[0., 0.])
                                
        self.classifier_visual = FlexibleMLP(input_size=1 * encoder_embed_dim,
                                    hidden_sizes=[2 * encoder_embed_dim,
                                                    int(1 * encoder_embed_dim)],
                                    num_classes=1,
                                    drop_rates=[0., 0.])

        self.AudioVisualInteractionModule = AudioVisualInteractionModule(num_layers=1)

        self.AudioTokenReducer_3 = TokenWise_TokenReducer()
        self.AudioTokenReducer_6 = TokenWise_TokenReducer()
        self.AudioTokenReducer_9 = TokenWise_TokenReducer()
        self.AudioTokenReducer_12 = TokenWise_TokenReducer()

        self.VisualTokenReducer_3 = TokenWise_TokenReducer()
        self.VisualTokenReducer_6 = TokenWise_TokenReducer()
        self.VisualTokenReducer_9 = TokenWise_TokenReducer()
        self.VisualTokenReducer_12 = TokenWise_TokenReducer()

        self.AudioTokenReducer_AVI = TokenWise_TokenReducer()
        self.VisualTokenReducer_AVI = TokenWise_TokenReducer()

        self.pool_v = LearnableWeightedPool(num_layers=4)
        self.pool_a = LearnableWeightedPool(num_layers=4)
    
    def split_cls_patch_tokens(self, emb, n_segments=8):
        B, L, D = emb.shape
        seg_len = L // n_segments
        emb = emb.view(B, n_segments, seg_len, D)  # [B, 8, 1+P, D]

        cls_tokens = emb[:, :, 0, :]      # [B, 8, D]
        patch_tokens = emb[:, :, 1:, :]    # [B, 8, P, D]

        return cls_tokens, patch_tokens.reshape(B, -1, D)

    def forward(self, audio=None, video=None, is_training=True):
        # audio: (B, 1, 1024, 128)
        # video: (B, 3, 16, 224, 224)

        # feature extraction
        video_3, video_6, video_9, video_12 = self.visual_encoder(video, ids_keep=None, apply_cls_tokens=True, use_mask=False)
        audio_3, audio_6, audio_9, audio_12 = self.audio_encoder(audio, ids_keep=None, apply_cls_tokens=True, use_mask=False)

        # remove cls tokens
        video_cls_tokens_3, video_3 = self.split_cls_patch_tokens(video_3, n_segments=8) 
        video_cls_tokens_6, video_6 = self.split_cls_patch_tokens(video_6, n_segments=8)
        video_cls_tokens_9, video_9 = self.split_cls_patch_tokens(video_9, n_segments=8)
        video_cls_tokens_12, video_12 = self.split_cls_patch_tokens(video_12, n_segments=8) # shape: (B, 8, 768), (B, 1568, 768)

        audio_cls_tokens_3, audio_3 = self.split_cls_patch_tokens(audio_3, n_segments=8)
        audio_cls_tokens_6, audio_6 = self.split_cls_patch_tokens(audio_6, n_segments=8)
        audio_cls_tokens_9, audio_9 = self.split_cls_patch_tokens(audio_9, n_segments=8)
        audio_cls_tokens_12, audio_12 = self.split_cls_patch_tokens(audio_12, n_segments=8)  # shape: (B, 8, 768), (B, 512, 768)

        video_cls_tokens_12 = video_cls_tokens_12.unsqueeze(dim=2)  # (B, 8, 1, 768)
        audio_cls_tokens_12 = audio_cls_tokens_12.unsqueeze(dim=2)  # (B, 8, 1, 768)

        # audio-visual interaction
        video_inter, audio_inter = self.AudioVisualInteractionModule(video_12, audio_12, video_cls_tokens = video_cls_tokens_12, audio_cls_tokens = audio_cls_tokens_12)
        
        _, video_inter = self.split_cls_patch_tokens(video_inter, n_segments=8) # shape: (B, 8, 768), (B, 1568, 768)    
        _, audio_inter = self.split_cls_patch_tokens(audio_inter, n_segments=8) # shape: (B, 8, 768), (B, 512, 768)     

        # adaptive aggregation
        aggregated_audio_3 = self.AudioTokenReducer_3(audio_3) # shape: [B, 768]
        aggregated_audio_6 = self.AudioTokenReducer_6(audio_6)
        aggregated_audio_9 = self.AudioTokenReducer_9(audio_9)
        aggregated_audio_12 = self.AudioTokenReducer_12(audio_12)
        aggregated_video_inter = self.VisualTokenReducer_AVI(video_inter)

        aggregated_video_3 = self.VisualTokenReducer_3(video_3) # shape: [B, 768]
        aggregated_video_6 = self.VisualTokenReducer_6(video_6)
        aggregated_video_9 = self.VisualTokenReducer_9(video_9)
        aggregated_video_12 = self.VisualTokenReducer_12(video_12)
        aggregated_audio_inter = self.AudioTokenReducer_AVI(audio_inter)

        audio_feats = torch.stack([
            aggregated_audio_3,
            aggregated_audio_6,
            aggregated_audio_9,
            aggregated_audio_12,
        ], dim=1)  # shape: [B, 4, 768]

        video_feats = torch.stack([
            aggregated_video_3,
            aggregated_video_6,
            aggregated_video_9,
            aggregated_video_12,
        ], dim=1)  # shape: [B, 4, 768]

        # Learnable weighted pooling over 3/6/9/12 encoder layers
        aggregated_audio_feats = self.pool_a(audio_feats)  # [B, 768]
        aggregated_video_feats = self.pool_v(video_feats)  # [B, 768]
        
        # Final feature: concat crossmodal + unimodal feats    
        final_feat = torch.cat([aggregated_audio_feats, aggregated_video_feats, aggregated_audio_inter, aggregated_video_inter], dim=1)  # [B, 4*768] 

        # overall classification
        overall_real_or_fake = self.classifier(final_feat)
    
        # auxiliary task: audio/visual classification
        if is_training:
            audio_real_or_fake = self.classifier_audio(aggregated_audio_feats)  # [B, 1]
            video_real_or_fake = self.classifier_visual(aggregated_video_feats)  # [B, 1]
            return audio_real_or_fake, video_real_or_fake, overall_real_or_fake
        else:
            return overall_real_or_fake

class LearnableWeightedPool(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, features: torch.Tensor):
        """
        features: Tensor of shape [B, num_layers, T, D]
        or [B, num_layers, D] — will handle both cases.
        """
        weights = F.softmax(self.weights, dim=0)  # (num_layers,)

        if features.dim() == 4:
            # [B, L, T, D] → weighted sum over L
            weighted = (weights[None, :, None, None] * features).sum(dim=1)  # [B, T, D]
        elif features.dim() == 3:
            # [B, L, D] → weighted sum over L
            weighted = (weights[None, :, None] * features).sum(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unsupported input shape: {features.shape}")
        return weighted

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, drop_rates=None, activation_fn=nn.ReLU):
        """
        Arguments:
        - input_size: input feature dimension, e.g., 768
        - hidden_sizes: list of hidden layer sizes, e.g., [512, 256]
        - num_classes: output dimension, e.g., 1 (binary classification)
        - drop_rates: list of Dropout rates for each layer, e.g., [0.1, 0.1] (must match the length of hidden_sizes)
        - activation_fn: activation function class (default: nn.ReLU, can be nn.LeakyReLU)
        """
        super(FlexibleMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()

        if drop_rates is None:
            drop_rates = [0.0] * len(hidden_sizes)
        assert len(drop_rates) == len(hidden_sizes), "drop_rates length must match hidden_sizes length"

        prev_size = input_size
        for hidden_size, drop_rate in zip(hidden_sizes, drop_rates):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(drop_rate))
            self.activations.append(activation_fn())
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for fc, drop, act in zip(self.layers, self.dropouts, self.activations):
            x = fc(x)
            x = drop(x)
            x = act(x)
        return self.output_layer(x)

def init_weights(m):
    """
    General function for weight initialization:
    - Apply Xavier uniform initialization to nn.Linear layers
    - Initialize biases to 0
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)