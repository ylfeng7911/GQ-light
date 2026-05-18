import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from ...core import register


__all__ = ["YOLO11Backbone"]


# =====================================
# Conv + BN + SiLU（官方 YOLO 默认）
# =====================================
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s,
            padding=k//2 if p is None else p,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =====================================
# YOLOv11 官方 C3k2（CSP Bottleneck）
#
# args: [c2, use_shortcut, width_multiple]
# =====================================
class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, use_shortcut=True, width_multiple=1.0, max_channels=512):
        super().__init__()
        # 1) 计算内部宽度（减小通道）
        c_ = int(c2 * width_multiple)
        # 限制内部通道不超过 max_channels
        c_ = min(c_, max_channels)

        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c1, c_, 1, 1)
        self.m = nn.Sequential(*[
            ConvBNAct(c_, c_, 3, 1) for _ in range(n)
        ])
        self.cv3 = ConvBNAct(2 * c_, c2, 1, 1)
        self.use_shortcut = use_shortcut

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


# =====================================
# 官方 SPPF
# =====================================
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)

    def forward(self, x):
        x_ = self.cv1(x)
        y1 = self.pool(x_)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x_, y1, y2, y3], dim=1))


# =====================================
# 官方 C2PSA（简化版位置/通道注意力）
# =====================================
class C2PSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = ConvBNAct(c1, c2, 1, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        return out * self.attn(out)


# =====================================
# YOLO11-s Backbone
# =====================================
@register()
class YOLO11Backbone(nn.Module):
    def __init__(self, return_idx: List[int] = [0,1,2,3],
                 depth_multiple: float = 0.5,
                 width_multiple: float = 0.5,
                 max_channels: int = 1024):
        """
        Official YOLO11‑s‑like backbone
        Args:
            return_idx: which feature maps to return (0=P2,1=P3,2=P4,3=P5)
            depth_multiple: official depth scale (1.0 for L)
            width_multiple: official width scale (1.0 for L)
            max_channels: intra‑C3 cap
        """
        super().__init__()
        self.return_idx = return_idx
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.max_channels = max_channels

        # ---------------------------------------------------
        # Stage 0 / 1: stem convs
        # ---------------------------------------------------
        self.conv0 = ConvBNAct(3, int(64 * width_multiple), 3, 2)
        self.conv1 = ConvBNAct(int(64 * width_multiple), int(128 * width_multiple), 3, 2)

        # ---------------------------------------------------
        # Stage 2: C3k2 → 256
        # ---------------------------------------------------
        c2 = int(256 * width_multiple)
        n2 = max(round(2 * depth_multiple), 1)
        self.c32 = C3k2(int(128 * width_multiple), c2, n=n2,
                        use_shortcut=False,
                        width_multiple=width_multiple,
                        max_channels=max_channels)

        # ---------------------------------------------------
        # Stage 3: C3k2 → 512
        # ---------------------------------------------------
        self.conv2 = ConvBNAct(c2, c2, 3, 2)
        c3 = int(512 * width_multiple)
        n3 = max(round(2 * depth_multiple), 1)
        self.c33 = C3k2(c2, c3, n=n3,
                        use_shortcut=False,
                        width_multiple=width_multiple,
                        max_channels=max_channels)

        # ---------------------------------------------------
        # Stage 4: C3k2 → 512
        # ---------------------------------------------------
        self.conv3 = ConvBNAct(c3, c3, 3, 2)
        n4 = max(round(2 * depth_multiple), 1)
        self.c34 = C3k2(c3, c3, n=n4,
                        use_shortcut=True,
                        width_multiple=width_multiple,
                        max_channels=max_channels)

        # ---------------------------------------------------
        # Stage 5: C3k2 → 1024
        # ---------------------------------------------------
        self.conv4 = ConvBNAct(c3, int(1024 * width_multiple), 3, 2)
        c5 = int(1024 * width_multiple)
        n5 = max(round(2 * depth_multiple), 1)
        self.c35 = C3k2(c5, c5, n=n5,
                        use_shortcut=True,
                        width_multiple=width_multiple,
                        max_channels=max_channels)

        # ---------------------------------------------------
        # SPPF + C2PSA
        # ---------------------------------------------------
        self.sppf = SPPF(c5, c5, k=5)
        self.c2psa = C2PSA(c5, c5)

        # Weight init
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        outs = []

        # P2, stride=4
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.c32(x)
        if 0 in self.return_idx:
            outs.append(x)

        # P3, stride=8
        x = self.conv2(x)
        x = self.c33(x)
        if 1 in self.return_idx:
            outs.append(x)

        # P4, stride=16
        x = self.conv3(x)
        x = self.c34(x)
        if 2 in self.return_idx:
            outs.append(x)

        # P5, stride=32
        x = self.conv4(x)
        x = self.c35(x)
        x = self.sppf(x)
        x = self.c2psa(x)
        if 3 in self.return_idx:
            outs.append(x)

        return outs
    
# 通道数[128, 256, 256, 512]