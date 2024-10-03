"""
By Shenyu Du
"""

import torch
import os
import torchvision
from typing import Optional, Tuple, Union, Dict, Any
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np


class TopQuery(nn.Module):
    def __init__(self,
                 fea_size: list,
                 query_k:
                 int = 300, ) -> None:
        super(TopQuery, self).__init__()

        self.query_k = query_k
        self.fea_size = fea_size
        in_dim = np.sum(self.fea_size)
        self.mlp = nn.Linear(in_dim, self.query_k, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape  # number:[B, 10000, 256]
        x = x.permute(0, 2, 1)
        print(x.shape)
        x1 = self.mlp(x)
        x1 = self.act(x1)

        x1 = x1.permute(0, 2, 1)
        print(x1.shape)
        return x1


class SIMDIS(nn.Module):
    def __init__(self,
                 fea_nums: list,
                 max_query:
                 int = 100) -> None:
        super(SIMDIS, self).__init__()
        self.fea_nums = fea_nums
        self.max_query = max_query

    def Simdis(self, x: Tensor, max_query: int) -> Tuple[Any, Any]:
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(0)

        #
        similarity_matrix = torch.nn.functional.cosine_similarity(x1, x2, dim=-1)
        # print(similarity_matrix.shape)

        #
        mean_similarity = torch.mean(similarity_matrix, dim=1)

        #
        # min_mean_similarity = torch.zeros(max_query)
        # for i in range(max_query):
        #    other_mean_similarity = torch.cat([mean_similarity[:i], mean_similarity[i + 1:]])
        #    min_mean_similarity[i] = torch.min(other_mean_similarity)

        #
        top_vectors = x[torch.argsort(mean_similarity)[:max_query]]

        # remember index
        max_index = torch.argsort(mean_similarity)[:max_query]  # fast max_query of index

        return top_vectors, max_index

    def forward(self, x):
        x, min_index = self.Simdis(x, self.max_query)

        return x, min_index


"""
By Shenyu Du
"""


# MTLAttention
class MultiHeadAttention(nn.Module):
    """
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        att_m(str): 'Aug' as the same feature map; 'Denoise' as not same; 'Mask' as background with foreground
        sparse(dict): example MLP param{0:2, 1:3, 2:4}
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            att_m: str,
            sparse: dict = {},
            attn_dropout: float = 0.0,
            bias: bool = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.att_m = att_m
        self.sparse = sparse
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] -> [N, h, P, P]
        attn = torch.matmul(query, key)

        if self.att_m == 'Aug':
            attn += 1
        elif self.att_m == 'Mask':
            attn -= 1000
        elif self.att_m == 'De':
            for k, v in self.sparse.items():
                attn -= v

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out


class DNAttention(nn.Module):
    def __init__(self,
                 tokens: list,
                 p_code: list,
                 attention=MultiHeadAttention):
        super(DNAttention, self).__init__()
        nums = len(tokens) + 1
        token_index = dict()

        for i in range(nums):
            token_index[i] = []

        self.Aug_att = attention
        self.De_att = attention
        self.Mask_att = attention
        self.tokens = tokens
        self.token_index = token_index
        self.p_code = p_code

    def forward(self, inputs, index):
        B, N, C = inputs.shape
        nums = len(self.tokens) + 1
        assert N == len(index), "N must == index numbers!"

        for id_p, id_x in enumerate(index):
            if 0 <= id_x < self.tokens[0]:
                self.token_index[0].append(id_p)
            elif self.tokens[0] <= id_x < self.tokens[1]:
                self.token_index[1].append(id_p)
            elif self.tokens[1] <= id_x < self.tokens[2]:
                self.token_index[2].append(id_p)
            else:
                self.token_index[3].append(id_p)

        output_aug = torch.ones_like(inputs)

        # Aug Attention
        ind = 0
        for k, v in self.token_index.items():
            nums = len(v)
            output_aug[:, ind:ind + nums, :] = self.Aug_att(embed_dim=256, num_heads=8, att_m='Aug', sparse={})(inputs[:, v, :]) - self.p_code[k]
            ind += nums


        # De Attention
        ind = 0
        for k, v in self.token_index.items():
            nums = len(v)
            output_aug[:, ind:ind + nums, :] = self.De_att(embed_dim=256, num_heads=8, att_m='De', sparse={})(
                inputs[:, v, :]) - self.p_code[k]
            ind += nums

        # Mask Attention
        ind = 0
        for k, v in self.token_index.items():
            nums = len(v)
            output_aug[:, ind:ind + nums, :] = self.Mask_att(embed_dim=256, num_heads=8, att_m='Mask', sparse={})(
                inputs[:, v, :]) - self.p_code[k]
            ind += nums

        # output_aug Tensor as Augmentation Attention
        output_aug = MultiHeadAttention(embed_dim=256, num_heads=8, att_m='default')(output_aug)

        return output_aug





