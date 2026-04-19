import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import to_2tuple, DropPath
from timm.models.vision_transformer import Mlp
from .positional_embedding import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
        proj_drop=0., attn_head_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):  # 加入 attn_mask 参数
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, H, N, N]

        if attn_mask is not None:
            # 注意力掩码应为 [1, 1, N, N] 或 [B, H, N, N]
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, audio_length=1024, mel_bins=128, patch_size=16, embed_dim=768, mlp_ratio=4.,
                 num_heads=12, encoder_depth=12, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., use_hierarchical=True):
        super().__init__()

        self.use_hierarchical = use_hierarchical
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, in_chans=1, patch_size=16, img_size=128)

        self.patch_embed.num_patches = int(audio_length * mel_bins / (patch_size ** 2))
        # nn.Parameter将一个张量注册为模型的一部分，并自动参与梯度计算和优化
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=True)
        self.modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_tokens = nn.Parameter(torch.zeros(1, 8, embed_dim), requires_grad=True)

        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 8, embed_dim), requires_grad=False)  # 非学习参数

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU)
            for _ in range(encoder_depth)])
        
        self.initialize_weights()

        self.initialize_cls_tokens()
    
    def forward_features(self, x, attn_mask=None, use_hierarchical=True):
        layer = 0
        features = []

        for block in self.blocks:
            layer += 1
            # 传入注意力掩码
            x = block(x, attn_mask=attn_mask)

            if use_hierarchical and (layer in [3, 6, 9, 12]):
                features.append(self.norm(x))

        if use_hierarchical:
            return tuple(features)
        else:
            x = self.norm(x)
            return x

    def apply_mask(self, x, ids_keep):
        index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x_masked = torch.gather(x, dim=1, index=index)  # [N, len_keep, D]
        return x_masked

    def forward(self, audio, ids_keep=None, apply_cls_tokens=True, use_hierarchical=None):

        if use_hierarchical is None:
            use_hierarchical = self.use_hierarchical

        # audio: (B, 1024, 128)
        audio = audio.unsqueeze(1)                # [B, 1, 1024, 128]
        audio = audio.permute(0, 1, 3, 2)          # [B, 1, 128, 1024]
        audio_emb = self.patch_embed(audio)       # [B, N_patches, D]
        audio_emb = audio_emb + self.pos_embed + self.modality

        if ids_keep is not None:
            audio_emb = self.apply_mask(audio_emb, ids_keep)

        if apply_cls_tokens:
            B = audio_emb.size(0)
            T = 8
            assert audio_emb.size(1) % T == 0, f"不能均分为 {T} 个时间片段"
            P = audio_emb.size(1) // T
            audio_emb = audio_emb.view(B, T, P, -1)                     # [B, 8, P, D]

            # 加上 cls_pos_embed：对每个 cls_token 加上对应位置编码
            cls_tokens = self.cls_tokens + self.cls_pos_embed  # [1, 8, D]
            cls_tokens = self.cls_tokens.expand(B, -1, -1)[:, :, None, :]  # [B, 8, 1, D]
            audio_emb = torch.cat([cls_tokens, audio_emb], dim=2)     # [B, 8, 1+P, D]
            audio_emb = audio_emb.reshape(B, -1, self.embed_dim)  # [B, T*(1+P), D]

        audio_emb = self.forward_features(audio_emb, attn_mask=None, use_hierarchical=use_hierarchical)
        return audio_emb
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 8, int(self.patch_embed.num_patches / 8), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.modality, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def initialize_cls_tokens(self):
        # 初始化 learnable cls tokens
        nn.init.normal_(self.cls_tokens, std=0.02)

        # 生成固定的 cls 位置编码
        cls_pos = np.arange(8)
        cls_pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, cls_pos)
        self.cls_pos_embed.data.copy_(torch.from_numpy(cls_pos_embed).unsqueeze(0))  # [1, 8, D]

# HierarchcalAttentionPooling fusion module in decoder
class HierarchcalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=6):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.encoder2decoder = nn.Linear(768, 384)
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        # Added normalization layers for stability
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        
        # 初始化参数
        self._init_weights()  

    def _init_weights(self):
        # 使用 Xavier 初始化，适合 transformer 中的投影层
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, query_feat, key_feat):
        # query_feat: (B, T1, D)
        # key_feat: (B, T2, D)
            
        B, T1, D = query_feat.shape
        T2 = key_feat.shape[1]
        key_feat = self.encoder2decoder(key_feat)  # (B, T2, D) -> (B, T2, D)

        # Normalize inputs
        query_feat = self.norm_q(query_feat)
        key_feat = self.norm_k(key_feat)

        Q = self.query_proj(query_feat)  # (B, T1, D)
        K = self.key_proj(key_feat)      # (B, T2, D)
        V = self.value_proj(key_feat)    # (B, T2, D)

        # Multi-head reshape
        Q = Q.view(B, T1, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, H, T1, d)
        K = K.view(B, T2, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, H, T2, d)
        V = V.view(B, T2, self.num_heads, D // self.num_heads).transpose(1, 2)  # (B, H, T2, d)

        # Attention weights: (B, H, T1, T2)
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)

        # Weighted sum: (B, H, T1, d)
        attn_output = attn_weights @ V

        # Merge heads: (B, T1, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, T1, D)

        # Residual connection and final norm
        return self.out_norm(query_feat + attn_output)

class AudioDecoder(nn.Module):
    def __init__(self, num_patches=(1024 * 128 // 256), patch_size=16, encoder_embed_dim=768, 
                 decoder_embed_dim=384, num_heads=6, decoder_depth=4, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., use_hierarchical=True):
        super().__init__()

        self.use_hierarchical = use_hierarchical
        self.num_patches = num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=True)  # learnable mask token
        self.decoder_modality = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU)
            for _ in range(decoder_depth)])
        
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True) # channel to 256
        self.decoder_pred_1 = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True) # channel to 256 
        self.decoder_pred_2 = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True) # channel to 256 
        self.decoder_pred_3 = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True) # channel to 256  
        
        self.HiA_3 = HierarchcalAttention(decoder_embed_dim, num_heads=6)
        self.HiA_6 = HierarchcalAttention(decoder_embed_dim, num_heads=6)
        self.HiA_9 = HierarchcalAttention(decoder_embed_dim, num_heads=6)
        self.HiA_12 = HierarchcalAttention(decoder_embed_dim, num_heads=6)

        self.initialize_weights()

    def hierarchical_addition(x, audio_layers, ids_keep_audio):
        """
        x: [N, L, D] decoder tokens
        audio_layers: list of encoder features [audio_3, audio_6, ...], each [N, L_keep, D]
        ids_keep_audio: [N, L_keep] 保留 token 的索引
        """
        N, L, D = x.shape

        for audio in audio_layers:
            # audio: [N, L_keep, D]
            for i in range(N):
                x[i, ids_keep_audio[i]] += audio[i]  # 直接加到对应位置
        return x

    def recover_from_mask(self, x_masked, ids_keep, mask_token, num_tokens=512):
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

    def forward(self, audio, audio_3=None, audio_6=None, audio_9=None, audio_12=None, ids_keep_audio=None):
        x = self.decoder_embed(audio)
        x = self.recover_from_mask(x, ids_keep_audio, self.mask_token)
        x = x + self.decoder_modality
        x = x + self.decoder_pos_embed

        if self.use_hierarchical:
            i = 1
            for blk in self.blocks:
                if i == 1:
                    x = self.HiA_12(x, audio_12)
                    x = self.HiA_9(x, audio_9)
                    x = self.HiA_6(x, audio_6)
                    x = self.HiA_3(x, audio_3)
                    x = blk(x)
                    x_1 = x
                elif i == 2:
                    x = self.HiA_9(x, audio_9)
                    x = self.HiA_6(x, audio_6)
                    x = self.HiA_3(x, audio_3)
                    x = blk(x)
                    x_2 = x
                elif i == 3:
                    x = self.HiA_6(x, audio_6)
                    x = self.HiA_3(x, audio_3)
                    x = blk(x)
                    x_3 = x
                elif i == 4:
                    x = self.HiA_3(x, audio_3)
                    x = blk(x)
                    x_4 = x
                i += 1
        else:
            for blk in self.blocks:
                x = blk(x)

        x_1 = self.decoder_pred_1(self.decoder_norm(x_1))
        x_2 = self.decoder_pred_2(self.decoder_norm(x_2))
        x_3 = self.decoder_pred_3(self.decoder_norm(x_3))
        x_4 = self.decoder_pred(self.decoder_norm(x_4))
        
        return x_1, x_2, x_3, x_4
        
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 8, int(self.num_patches/8), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.decoder_modality, std=.02)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)