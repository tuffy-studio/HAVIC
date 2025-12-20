import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .positional_embedding import *
from .utils import no_grad_trunc_normal_
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
import torch.nn.init as init


class SelfAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class CrossModalAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # Normalization layers
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Feed-forward network (FFN)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, query_feat, key_feat, use_re_softmax=True):
        B, Tq, D = query_feat.shape
        Tk = key_feat.shape[1]

        # Normalize before projections
        q = self.norm_q(query_feat)
        k = self.norm_k(key_feat)
        v = key_feat

        # Project Q, K, V
        q = self.q_proj(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if use_re_softmax:
            attn = torch.softmax(-attn, dim=-1)  # re-softmax → 强调互补信息
        else:
            attn = torch.softmax(attn, dim=-1)   # 普通 softmax → 强调相似性
        #attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, Tq, D)
        out = self.proj_drop(self.proj(out))

        # Residual connection 1
        x = query_feat + self.drop_path(out)

        # FFN + Residual connection 2
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
 
class AVIMBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 自注意力层（分别对video和audio）
        self.self_attn_video = SelfAttention(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.self_attn_audio = SelfAttention(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

        # 跨模态交叉注意力层
        self.cross_attn_audio_to_video = CrossModalAttention(dim=dim, num_heads=num_heads,
                                                             qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop)
        self.cross_attn_video_to_audio = CrossModalAttention(dim=dim, num_heads=num_heads,
                                                             qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop)

    def forward(self, video, audio, local=False, time_segments=8, neighbor_window=1):

        # 1) 跨模态特征补充
        if local == False:
            video_inter = video + self.cross_attn_audio_to_video(video, audio, use_re_softmax=False)
            audio_inter = audio + self.cross_attn_video_to_audio(audio, video, use_re_softmax=False)

        else:
            # 单帧分割
            if neighbor_window ==0:
                B, N, D = video.shape
                video_split = video.reshape(B, time_segments, -1, D).reshape(B*time_segments, -1, D)
                audio_split = audio.reshape(B, time_segments, -1, D).reshape(B*time_segments, -1, D)

                video_inter = self.cross_attn_audio_to_video(video_split, audio_split).reshape(B, time_segments, -1, D).reshape(B, -1, D)
                audio_inter = self.cross_attn_video_to_audio(audio_split, video_split).reshape(B, time_segments, -1, D).reshape(B, -1, D)
            else:
                B, N, D = video.shape
                video_split = video.contiguous().reshape(B, time_segments, -1, D)
                audio_split = audio.contiguous().reshape(B, time_segments, -1, D)
                video_inter_list, audio_inter_list = [], []

                for t in range(time_segments):
                    # 计算邻域范围
                    start = max(0, t - neighbor_window)
                    end = min(time_segments, t + neighbor_window + 1)

                    # 获取局部上下文帧
                    audio_context = audio_split[:, start:end].reshape(B, -1, D)
                    video_context = video_split[:, start:end].reshape(B, -1, D)

                    # 当前帧特征
                    video_t = video_split[:, t]
                    audio_t = audio_split[:, t]

                    # 跨模态注意力（当前帧 <-> 邻域音频）
                    video_inter_t = self.cross_attn_audio_to_video(video_t, audio_context)
                    audio_inter_t = self.cross_attn_video_to_audio(audio_t, video_context)

                    video_inter_list.append(video_inter_t)
                    audio_inter_list.append(audio_inter_t)

                # 拼接回完整序列
                video_inter = torch.cat(video_inter_list, dim=1)
                audio_inter = torch.cat(audio_inter_list, dim=1)

        # 2) 自注意力机制汇集信息
        video = self.self_attn_video(video)
        audio = self.self_attn_audio(audio)

        return video_inter, audio_inter

class AudioVisualInteractionModule(nn.Module):
    def __init__(self, num_layers=2, dim=768, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.dim = dim
        # 音频
        self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=True)  # learnable mask token
        self.audio_modality = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=True)
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, 512, 768), requires_grad=False)

        # 视频
        self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        no_grad_trunc_normal_(self.visual_mask_token, mean=0., std=0.02, a=-0.02, b=0.02)
        self.visual_pos_embedding = SinCosPositionalEmbedding((1568, 768), dropout_rate=0.)
        
        self.blocks = nn.ModuleList([
            AVIMBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])

        self.norm_video_out = nn.LayerNorm(dim)
        self.norm_audio_out = nn.LayerNorm(dim)

        self.initialize_weights()

    def forward(self, video, audio, video_cls_tokens=None, audio_cls_tokens=None):

        # 插入cls token：
        if video_cls_tokens is not None and audio_cls_tokens is not None:

            B = audio.size(0)
            T = 8
            assert audio.size(1) % T == 0, f"不能均分为 {T} 个时间片段"
            P = audio.size(1) // T
            audio = audio.view(B, T, P, -1)                     # [B, 8, P, D]
            audio = torch.cat([audio_cls_tokens, audio], dim=2)     # [B, 8, 1+P, D]
            audio = audio.reshape(B, -1, self.dim)  # [B, T*(1+P), D]

            assert video.shape[1] % T == 0, f"不能均分为 {T} 个时间片段"
            P = video.shape[1] // T  # 每帧 patch 数
            video = video.view(B, T, P, -1) 
            video = torch.cat([video_cls_tokens, video], dim=2)  # [B, 8, 1+P, D]
            video = video.reshape(B, -1, self.dim)  # [B, N, D]，N = 8 * (1 + P)

        # GLAVI = True
        # if GLAVI == False: # ablation study without cross-modal interaction
        #     return video, audio
        
        for blk in self.blocks:
            video, audio = blk(video, audio)

        # 在输出前加norm
        video = self.norm_video_out(video)
        audio = self.norm_audio_out(audio)
        
        return video, audio


    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(768, 8, 64, cls_token=False)
        self.audio_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.audio_modality, std=.02)
        
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

def init_weights(m):
    """
    初始化权重的通用函数：
    - 对 nn.Linear 层使用 Xavier uniform 初始化
    - 对 bias 使用 0 初始化
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes,
                 drop_rates=None, activation_fn=nn.ReLU, use_layernorm=False):
        """
        参数说明：
        - input_size: 输入特征维度，如 768
        - hidden_sizes: 隐藏层大小列表，如 [512, 256]
        - num_classes: 输出维度，如 1（二分类）
        - drop_rates: 每层的 Dropout 概率列表，如 [0.1, 0.1]（长度必须与 hidden_sizes 相同）
        - activation_fn: 激活函数类（默认是 nn.ReLU，可传 nn.LeakyReLU）
        - use_layernorm: 是否使用 LayerNorm（默认 False）
        """
        super(FlexibleMLP, self).__init__()

        self.use_layernorm = use_layernorm
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()

        if drop_rates is None:
            drop_rates = [0.0] * len(hidden_sizes)
        assert len(drop_rates) == len(hidden_sizes), "drop_rates 和 hidden_sizes 长度不一致"

        prev_size = input_size
        for hidden_size, drop_rate in zip(hidden_sizes, drop_rates):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_layernorm:
                self.layernorms.append(nn.LayerNorm(hidden_size))
            else:
                self.layernorms.append(None)
            self.dropouts.append(nn.Dropout(drop_rate))
            self.activations.append(activation_fn())
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for fc, norm, drop, act in zip(self.layers, self.layernorms, self.dropouts, self.activations):
            x = fc(x)
            if self.use_layernorm and norm is not None:
                x = norm(x)
            x = drop(x)
            x = act(x)
        return self.output_layer(x)

class A2V_Decoder(nn.Module):
    def __init__(self, audio_dim=64, visual_dim=196, embed_dim=768, num_heads=12):
        super(A2V_Decoder, self).__init__()
        self.TokenAligner = FlexibleMLP(input_size=audio_dim, hidden_sizes=[], num_classes=visual_dim, activation_fn=nn.GELU)
        self.block = SelfAttention(num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=True)  # learnable mask token
        self.audio_modality = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=True)
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, 512, 768), requires_grad=False)

        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(768, 8, 64, cls_token=False)
        self.audio_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.audio_modality, std=.02)
        
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

    def forward(self, audio, time_frames=8, ids_keep_audio=None):
        # audio.shape: [B, 512, 768]
        audio = self.recover_from_mask(audio, ids_keep_audio, self.audio_mask_token, num_tokens=512)
        audio = audio + self.audio_modality
        audio = audio + self.audio_pos_embed

        B, N, D = audio.shape
        audio = audio.reshape(B, time_frames, -1, D) # [B, T, 512/T, 768], T=8
        audio = audio.reshape(B*time_frames, -1 ,D).transpose(1,2) # [B*T, 768, 512/T], T=8
        
        video_tokens = self.TokenAligner(audio).transpose(1,2) # [B*T, 1568/T, 768], T=8
        
        trans_video_emb = self.block(video_tokens.reshape(B,-1,D)) # [B, 1568, 768]

        return self.norm(trans_video_emb)

class V2A_Decoder(nn.Module):
    def __init__(self, audio_dim=64, visual_dim=196, embed_dim=768, num_heads=12):
        super(V2A_Decoder, self).__init__()
        self.TokenAligner = FlexibleMLP(input_size=visual_dim, hidden_sizes=[], num_classes=audio_dim, activation_fn=nn.GELU)
        self.block = SelfAttention(num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        no_grad_trunc_normal_(self.visual_mask_token, mean=0., std=0.02, a=-0.02, b=0.02)
        self.visual_pos_embedding = SinCosPositionalEmbedding((1568, 768), dropout_rate=0.)

        self.initialize_weights()

    def initialize_weights(self):
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
        
    def forward(self, video, time_frames=8, ids_keep_video=None):
        # video.shape: [B, 1568, 768]
        video = self.recover_from_mask(video, ids_keep_video, self.visual_mask_token, num_tokens=1568)
        video = self.visual_pos_embedding(video)

        B, N, D = video.shape
        video = video.reshape(B, time_frames, -1, D) # [B, T, 1568/T, 768], T=8
        video = video.reshape(B*time_frames, -1 ,D).transpose(1,2) # [B*T, 768, 1568/T], T=8
        
        audio_tokens = self.TokenAligner(video).transpose(1,2) # [B*T, 512/T, 768], T=8
        
        trans_audio_emb = self.block(audio_tokens.reshape(B, -1, D))

        return self.norm(trans_audio_emb)