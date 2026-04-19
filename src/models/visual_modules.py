import torch
from einops import rearrange
from torch import nn, Tensor
import numpy as np
from .utils import PatchEmbedding3d, Block3d, no_grad_trunc_normal_
from .positional_embedding import SinCosPositionalEmbedding, get_1d_sincos_pos_embed_from_grid

class VisualEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, n_frames=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2, use_hierarchical=True
    ):
        super().__init__()
        self.use_hierarchical = use_hierarchical
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding3d(
            input_size=(3, n_frames, img_size, img_size),
            patch_size=(tubelet_size, patch_size, patch_size),
            embedding=embed_dim
        )
        num_patches = (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)
        
        self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.)
        
        if norm_layer == 'LayerNorm':
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError(f"Normalization layer {norm_layer} not implemented")
        
        self.blocks = nn.ModuleList([
            Block3d(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])

        self.cls_tokens = nn.Parameter(torch.zeros(1, 8, embed_dim), requires_grad=True) # learnable contrastive time token

        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 8, embed_dim), requires_grad=False)  # 非学习参数
        
        self.apply(self._init_weights)

        self.initialize_cls_tokens()
        
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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
        x_masked = torch.gather(x, dim=1, index=index)  # [B, len_keep, D]
        return x_masked
    
    def forward(self, x, ids_keep=None, apply_cls_tokens=True, use_hierarchical=None):

        if use_hierarchical is None:
            use_hierarchical = self.use_hierarchical
            
        assert len(x.shape) == 5, "x must be 5D: B, C, T, H, W"
        B = x.shape[0]

        emb = self.patch_embedding(x)   # [B, L=1568, D]
        emb = self.pos_embedding(emb) 
        if ids_keep is not None:
            emb = self.apply_mask(emb, ids_keep)

        # === 插入 cls tokens，并构造 attn_mask（可选） ===
        if apply_cls_tokens:
            T = 8
            assert emb.shape[1] % T == 0, f"不能均分为 {T} 个时间片段"
            P = emb.shape[1] // T  # 每帧 patch 数

            emb = emb.reshape(B, T, P, self.embed_dim)

            # 插入 cls tokens
            cls_tokens = self.cls_tokens + self.cls_pos_embed  # [1, 8, D]
            cls_tokens = cls_tokens.expand(B, -1, -1)[:, :, None, :]  # [B, 8, 1, D]
            emb = torch.cat([cls_tokens, emb], dim=2)  # [B, 8, 1+P, D]
            emb = emb.reshape(B, -1, self.embed_dim)  # [B, N, D]，N = 8 * (1 + P)

        # === 进入 transformer blocks，带 attn_mask ===
        emb = self.forward_features(emb, attn_mask=None, use_hierarchical=self.use_hierarchical)
        return emb

# HierarchcalAttentionPooling fusion module in decoder
class HierarchcalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=6):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.encoder2decoder = nn.Linear(768, 384)

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
        key_feat = self.encoder2decoder(key_feat)  # (B, T2, D)

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

class VisualDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, n_frames=16, embed_dim=384, depth=8,
        num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=1., tubelet_size=2, encoder_embed_dim=768, use_hierarchical=True
    ):
        super().__init__()
        output_dim = 3 * tubelet_size * patch_size * patch_size
        self.use_hierarchical = use_hierarchical
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.n_patch_h = img_size // patch_size
        self.n_patch_w = img_size // patch_size
        self.embed_dim = embed_dim
        if norm_layer == "LayerNorm":
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        self.enc_dec_proj = nn.Linear(encoder_embed_dim, embed_dim, bias=False)

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding(
            (self.n_patch_h * self.n_patch_w * (n_frames // tubelet_size), embed_dim), dropout_rate=0.)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        no_grad_trunc_normal_(self.mask_token, mean=0., std=0.02, a=-0.02, b=0.02)

        self.blocks = nn.ModuleList([
            Block3d(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values
            ) for _ in range(depth)])

        self.head = nn.Linear(embed_dim, output_dim)
        self.head_1 = nn.Linear(embed_dim, output_dim)
        self.head_2 = nn.Linear(embed_dim, output_dim)
        self.head_3 = nn.Linear(embed_dim, output_dim)
        self.apply(self._init_weights)
        

        self.HiA_3 = HierarchcalAttention(embed_dim, num_heads=6)
        self.HiA_6 = HierarchcalAttention(embed_dim, num_heads=6)
        self.HiA_9 = HierarchcalAttention(embed_dim, num_heads=6)
        self.HiA_12 = HierarchcalAttention(embed_dim, num_heads=6)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def recover_from_mask(self, x_masked, ids_keep, mask_token, num_tokens=1568):
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

    def unpatch_to_img(self, x: Tensor) -> Tensor:
        # x: (Batch_size, token_nums, 16*16*3*2)
        x = rearrange(x, "b n (c p) -> b n p c", c=3) 

        # x: (Batch_size, token_nums, 16*16*2, C)
        x = rearrange(x, "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)", p0=self.tubelet_size,
            p1=self.patch_size, p2=self.patch_size, h=self.n_patch_h, w=self.n_patch_w)
        # x: (B, C, T, H, W)
        return x


    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return x

    def forward(self, x, video_3=None, video_6=None, video_9=None, video_12=None, ids_keep_video=None):
        x = self.enc_dec_proj(x)
        x = self.recover_from_mask(x, ids_keep_video, self.mask_token)
        x = self.pos_embedding(x)
        if self.use_hierarchical:
            i=1
            for blk in self.blocks:
                if i == 1:
                    x = self.HiA_12(x, video_12)
                    x = self.HiA_9(x, video_9)
                    x = self.HiA_6(x, video_6)
                    x = self.HiA_3(x, video_3)
                    x = blk(x)
                    x_1 = x
                elif i == 2:
                    x = self.HiA_9(x, video_9)
                    x = self.HiA_6(x, video_6)
                    x = self.HiA_3(x, video_3)
                    x = blk(x)
                    x_2 = x
                elif i == 3:
                    x = self.HiA_6(x, video_6)
                    x = self.HiA_3(x, video_3)
                    x = blk(x)
                    x_3 = x
                elif i == 4:
                    x = self.HiA_3(x, video_3)
                    x = blk(x)
                    x_4 = x
                i += 1

            x_1 = self.head_1(self.norm(x_1))
            x_2 = self.head_2(self.norm(x_2))
            x_3 = self.head_3(self.norm(x_3))
            x_4 = self.head(self.norm(x_4))
            return x_1, x_2, x_3, x_4
        else:
            x = self.forward_features(x)
            return x