"""
    By Shenyu Du
    Encoder strategy two method as Hierarchy Feature Encoder Module
"""

import torch
import os
import torchvision
from typing import Optional, Tuple, Union, Dict, Any
import math
import configparser
from TransformerEncoder import Attention
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np


class AdjustWindowSize(nn.Module):
    """
    Batchsize must be 1;
    Args:
        down_m: Denotes the down sampling multiple of the feature map to the original images;
        gt_boxes: Dictionary sorted by target box size;
        window_size: This feature map's size as [H: int, W: int];
        offset_size: Window size after offsetting;
        sort: 1, 2, 3 represent feature map levels;
    """

    def __init__(self,
                 down_m: int,
                 gt_boxes: dict,
                 # window_size: int,
                 offset_size: int,
                 sort: int):
        super().__init__(AdjustWindowSize, self)
        self.down_m = down_m
        # self.window_size = window_size
        self.off = offset_size
        self.sort = sort
        k_num = 0

        for k, v in gt_boxes['GT_boxes'].items():
            k_num += 1

        if k_num >= 3:
            if self.sort == 1:
                self.gt_boxes = gt_boxes['GT_boxes']['0']
            elif self.sort == 2:
                self.gt_boxes = gt_boxes['GT_boxes']['1']
            else:
                self.gt_boxes = gt_boxes['GT_boxes']['2']
        elif k_num == 2:
            if self.sort == 1:
                self.gt_boxes = gt_boxes['GT_boxes']['0']
            else:
                self.gt_boxes = gt_boxes['GT_boxes']['1']
        else:
            self.gt_boxes = gt_boxes['GT_boxes']['0']

        # 计算目标映射得到特征图上的位置，以及需要调整到的位置
        self.gt_boxes_f = [int(x / self.down_m) for x in self.gt_boxes]

        # 调整 H，W
        self.H_off = abs(self.gt_boxes_f[3] - self.gt_boxes_f[1])
        self.W_off = abs(self.gt_boxes_f[2] - self.gt_boxes_f[0])


        if self.off > self.H_off and self.off > self.W_off:
            padding_H = self.off - self.H_off
            padding_W = self.off - self.W_off

            self.boxes_f2 = self.gt_boxes_f
            if self.gt_boxes_f[1] - padding_H <= 0:
                self.boxes_f2[3] += padding_H
            else:
                self.boxes_f2[1] -= padding_H
            if self.gt_boxes_f[0] - padding_W <= 0:
                self.boxes_f2[2] += padding_W
            else:
                self.boxes_f2[0] -= padding_W

    def forward(self, x: Tensor):
        B, C, H, W = x.shape

        x = x[..., self.boxes_f2[1]:self.boxes_f2[3] + 1, self.boxes_f2[0]:self.boxes_f2[2] + 1]

        # x_gt = x[..., self.gt_boxes_f[1]:self.gt_boxes_f[3] + 1, self.gt_boxes_f[0]:self.gt_boxes_f[2] + 1]

        # x_gt = self.inter(x_gt)

        return x


class HFEModuleBlock(nn.Module):
    """
    Args:
        blocks: int as features numbers;
        down_m: Denotes the down sampling multiple of the feature map to the original images;
        gt_boxes: Dictionary sorted by target box size;
        sort: 1, 2, 3 represent feature map levels;
        window_size: [size1, size2, ..., ] as feature maps size;
        adjust_block: AdjustWindowSize(nn.Module);
        attention: Multi-Head attention block;
        out_dimension: out dimensions default as 256;
    """

    def __init__(self,
                 down_m: list, #[8,16,32,64]
                 gt_boxes: dict,
                 sort: list,
                 # window_size: list,
                 blocks: int = 4,
                 out_dimension: int = 256,
                 adjust_block: Optional[nn.Module] = AdjustWindowSize,
                 attention: Optional[nn.Module] = Attention,
                 norm_layer: Optional[nn.Module] = nn.LayerNorm) -> None:
        super().__init__(HFEModuleBlock, self)
        self.down_m = down_m
        self.gt_boxes = gt_boxes
        self.sort = sort
        # self.window = window_size
        self.block = blocks
        self.adjust_block = adjust_block
        self.attention = attention
        self.norm = norm_layer
        self.out_dim = out_dimension
        self.mlp = nn.Linear

    def forward_x(self, p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor) -> Tensor:
        # For p4:
        B4, C4, H4, W4 = p4.shape  # -> B, L , P

        assert H4 == W4, f"Feature map size must be equal to H = W! But got {H4} and {W4}."
        offset_size = H4

        p4 = p4.flatten(2).transpose(1, 2)

        p4 = self.norm(C4)(p4) if self.norm is not nn.Identity else self.norm()(p4)

        p4_a = self.attention(dim=C4)(p4)

        # For p1:
        B1, C1, H1, W1 = p1.shape

        assert H1 == W1, f"Feature map size must be equal to H = W! But got {H1} and {W1}."

        p1_a = self.adjust_block(self.down_m[0], self.gt_boxes, offset_size, self.sort[0])(p1)

        p1_a = p1_a.flatten(2).transpose(1, 2)

        p1_a = self.norm(C1)(p1_a) if (self.norm is not nn.Identity) else self.norm()(p1_a)

        p1_a = self.attention(dim=C1)(p1_a)

        # For p2:
        B2, C2, H2, W2 = p2.shape

        assert H2 == W2, f"Feature map size must be equal to H = W! But got {H2} and {W2}."

        p2_a = self.adjust_block(self.down_m[1], self.gt_boxes, offset_size, self.sort[1])(p2)

        p2_a = p2_a.flatten(2).transpose(1, 2)

        p2_a = self.norm(C2)(p2_a) if (self.norm is not nn.Identity) else self.norm()(p2_a)

        p2_a = self.attention(dim=C2)(p2_a)

        # For p3:
        B3, C3, H3, W3 = p3.shape

        assert H3 == W3, f"Feature map size must be equal to H = W! But got {H3} and {W3}."

        p3_a = self.adjust_block(self.down_m[2], self.gt_boxes, offset_size, self.sort[2])(p3)

        p3_a = p3_a.flatten(2).transpose(1, 2)

        p3_a = self.norm(C3)(p3_a) if (self.norm is not nn.Identity) else self.norm()(p3_a)

        p3_a = self.attention(dim=C3)(p3_a)

        # For fusion p1 p2 p3 p4 :
        out = torch.cat((p1_a, p2_a, p3_a, p4_a), dim=2)
        B, N, dim = out.shape
        if dim == self.out_dim:
            return out
        out = self.mlp(dim, self.out_dim, bias=False)
        return out

    def forward(self, p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor) -> Tensor:
        out = self.forward_x(p1, p2, p3, p4)

        return out


if __name__ == "__main__":
    down_m2 = [2, 4, 8, 16]
    gt = {'GT_boxes': {'0': [], '1': [], '2': [], '3': []}}
    sort2 = [1, 2, 3, 4]
    hfemblock = HFEModuleBlock()
