import torch
from torch import Tensor, nn

from .utils import Shape
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_w_size, grid_h_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    生成严格交错的1D sin-cos位置编码。

    参数：
    embed_dim: int，位置编码的维度（必须为偶数）
    pos: numpy array，形状为 (M,) 的位置序列

    返回：
    emb: numpy array，形状为 (M, embed_dim)，位置编码
         位置维度为 [sin, cos, sin, cos, ...] 交替排列
    """
    assert embed_dim % 2 == 0, "embed_dim必须为偶数"

    half_dim = embed_dim // 2
    omega = np.arange(half_dim, dtype=np.float32)
    omega /= half_dim
    omega = 1.0 / (10000 ** omega)  # (half_dim,)

    pos = pos.reshape(-1, 1)  # (M, 1)
    angles = pos * omega      # (M, half_dim)

    emb = np.zeros((pos.shape[0], embed_dim), dtype=np.float32)
    emb[:, 0::2] = np.sin(angles)  # 偶数维度：sin
    emb[:, 1::2] = np.cos(angles)  # 奇数维度：cos

    return emb

class PositionalEmbedding(nn.Module):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value

class SinCosPositionalEmbedding(PositionalEmbedding):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()