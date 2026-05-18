"""
FasterNet backbone for RT-DETR / GQ-DETR.

Adapted from FasterNet-style implementation:
- Remove classification head: avg_pool / conv1 / fc
- Return multi-scale feature maps for detection
- Add @register() for YAMLConfig registry
- Replace external DropPath with local PyTorch implementation

Input:
    x: [B, 3, H, W]

Default output with return_idx=[1, 2, 3]:
    outs[0]: stride=8
    outs[1]: stride=16
    outs[2]: stride=32

If return_idx=[0, 1, 2, 3]:
    outs[0]: stride=4
    outs[1]: stride=8
    outs[2]: stride=16
    outs[3]: stride=32
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from .common import FrozenBatchNorm2d
from ...core import register


__all__ = ["FasterNet", "FasterNetBlock"]

class DropPath(nn.Module):
    """
    DropPath, also called Stochastic Depth.

    During training, randomly drops residual branches per sample.
    During eval, it becomes identity.

    Input:
        x: Tensor with shape [B, C, H, W] or other shapes
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Work with tensors of any dimension.
        # For image features [B, C, H, W], shape = [B, 1, 1, 1].
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


# =========================================================
# Partial Convolution
#
# FasterNet 的核心模块之一。
# 它只对一部分通道做 3x3 卷积，剩余通道保持不变。
#
# 目的：
#   减少冗余空间计算，提高真实推理效率。
# =========================================================
class PartialConv(nn.Module):
    def __init__(
        self,
        dim: int,
        n_div: int = 4,
        kernel_size: int = 3,
        forward_type: str = "split_cat",
    ):
        super().__init__()

        self.dim_conv3 = dim // n_div                   # 只有1/4通道参与卷积
        self.dim_untouched = dim - self.dim_conv3

        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        if forward_type == "slicing":
            self.forward = self.forward_slicing
        elif forward_type == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(
                f"Unsupported PartialConv forward_type: {forward_type}"
            )

    def forward_slicing(self, x: Tensor) -> Tensor:
        """
        Mainly for inference.
        注意：这里 clone 是为了避免破坏 residual 分支中的原始输入。
        """
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(
            x[:, : self.dim_conv3, :, :]
        )
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        """
        For training and inference.
        更安全，默认使用这个。
        """
        x1, x2 = torch.split(
            x, [self.dim_conv3, self.dim_untouched], dim=1
        )
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        return x


# =========================================================
# FasterNet Block
#
# Structure:
#   x
#   -> PartialConv
#   -> 1x1 Conv expand
#   -> BN
#   -> Act
#   -> 1x1 Conv project
#   -> DropPath
#   -> residual add
#
# 与 ResNet block 不同：
#   它不是完整 3x3 标准卷积，而是只对部分通道做空间卷积。
# =========================================================
class FasterNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 2,
        act_layer: nn.Module = nn.ReLU,
        drop_path_rate: float = 0.0,
        pconv_n_div: int = 4,
        pconv_forward_type: str = "split_cat",
    ):
        super().__init__()

        hidden_dim = dim * expand_ratio

        self.pconv = PartialConv(
            dim=dim,
            n_div=pconv_n_div,
            kernel_size=3,
            forward_type=pconv_forward_type,
        )

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)

        x = residual + self.drop_path(x)
        return x


# =========================================================
# FasterNet Backbone for Detection
#
# 输入:
#   x: [B, 3, H, W]
#
# 输出:
#   outs: List[Tensor]
#
# 默认 return_idx=[1,2,3] 时:
#   outs[0]: stride=8
#   outs[1]: stride=16
#   outs[2]: stride=32
#
# 如果 return_idx=[0,1,2,3]:
#   outs[0]: stride=4
#   outs[1]: stride=8
#   outs[2]: stride=16
#   outs[3]: stride=32
# =========================================================
@register()
class FasterNet(nn.Module):
    arch_configs = {
        # name: embed_dim, depths, act, default_drop_path_rate
        "t0": {
            "embed_dim": 40,
            "depths": [1, 2, 8, 2],
            "act": "gelu",
            "drop_path_rate": 0.0,
        },
        "t1": {
            "embed_dim": 64,
            "depths": [1, 2, 8, 2],
            "act": "gelu",
            "drop_path_rate": 0.02,
        },
        "t2": {
            "embed_dim": 96,
            "depths": [1, 2, 8, 2],
            "act": "relu",
            "drop_path_rate": 0.05,
        },
        "s": {
            "embed_dim": 128,
            "depths": [1, 2, 13, 2],
            "act": "relu",
            "drop_path_rate": 0.03,
        },
        "m": {
            "embed_dim": 144,
            "depths": [3, 4, 18, 3],
            "act": "relu",
            "drop_path_rate": 0.05,
        },
        "l": {
            "embed_dim": 192,
            "depths": [3, 4, 18, 3],
            "act": "relu",
            "drop_path_rate": 0.05,
        },
    }

    def __init__(
        self,
        name: str = "t0",
        in_channels: int = 3,
        return_idx: List[int] = [1, 2, 3],
        expand_ratio: int = 2,
        pconv_n_div: int = 4,
        pconv_forward_type: str = "split_cat",
        drop_path_rate: Optional[float] = None,     # 0.0
        freeze_at: int = -1,
        freeze_norm: bool = False,
        pretrained=False,
    ):
        super().__init__()

        if name not in self.arch_configs:
            raise ValueError(
                f"Unsupported FasterNet name={name}. "
                f"Available names: {list(self.arch_configs.keys())}"
            )

        cfg = self.arch_configs[name]
        embed_dim = cfg["embed_dim"]            # 64
        depths = cfg["depths"]                  # [1,2,8,2]

        if drop_path_rate is None:
            drop_path_rate = cfg["drop_path_rate"]

        if cfg["act"] == "gelu":
            act_layer = nn.GELU
        elif cfg["act"] == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError(f"Unsupported act type: {cfg['act']}")

        self.name = name
        self.return_idx = return_idx
        self.depths = depths
        self.embed_dim = embed_dim

        # -------------------------------------------------
        # Stem
        #
        # 输入:  [B, 3, H, W]
        # 输出:  [B, embed_dim, H/4, W/4]
        #
        # FasterNet 原始实现第一层就是 stride=4 patch embedding。
        # 所以它没有 stride=2 的输出层。
        # -------------------------------------------------
        self.stem = nn.Sequential(                  # r18的stem是3个卷积串联，s=2
            nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            act_layer(),
        )

        # -------------------------------------------------
        # Four stages.
        #
        # stage0: stride=4,  channels=embed_dim
        # stage1: stride=8,  channels=embed_dim*2
        # stage2: stride=16, channels=embed_dim*4
        # stage3: stride=32, channels=embed_dim*8
        # -------------------------------------------------
        total_blocks = sum(depths)
        drop_path_list = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, total_blocks)            
        ]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_dim = embed_dim
        block_idx = 0

        for stage_idx, depth in enumerate(depths):
            blocks = []

            # 构造[1,2,8,2]个Block
            for _ in range(depth):
                blocks.append(
                    FasterNetBlock(
                        dim=cur_dim,
                        expand_ratio=expand_ratio,              # 2
                        act_layer=act_layer,
                        drop_path_rate=drop_path_list[block_idx],
                        pconv_n_div=pconv_n_div,                # 4
                        pconv_forward_type=pconv_forward_type,  # "split_cat"
                    )
                )
                block_idx += 1

            self.stages.append(nn.Sequential(*blocks))

            # 前3个stage有 （最后一个stage没有）
            if stage_idx < len(depths) - 1:
                # Downsample between stages.
                # [B, C, H, W] -> [B, 2C, H/2, W/2]
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(
                            cur_dim,
                            cur_dim * 2,
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(cur_dim * 2),
                        act_layer(),
                    )
                )
                cur_dim *= 2

        self.out_channels = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
        ]
        self.out_strides = [4, 8, 16, 32]
        
        self._init_weights()

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at, len(self.stages))):
                self._freeze_parameters(self.stages[i])
                if i < len(self.downsamples):
                    self._freeze_parameters(self.downsamples[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, str):
                state = torch.load(pretrained, map_location="cpu")
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                self.load_state_dict(state, strict=False)
                print(f"Load FasterNet-{name} state_dict from {pretrained}")
            else:
                raise ValueError(
                    "pretrained=True is not supported because no official "
                    "download URL is provided in this local implementation. "
                    "Please pass a checkpoint path string or set pretrained=False."
                )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                new_child = self._freeze_norm(child)
                if new_child is not child:
                    setattr(m, name, new_child)
        return m

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Input:
            x: [B, 3, H, W]

        Output:
            outs: list of feature maps selected by return_idx.

        For input [B, 3, 640, 640], FasterNet-t0:
            stage0: [B, 40, 160, 160], stride=4
            stage1: [B, 80, 80, 80],   stride=8
            stage2: [B, 160, 40, 40],  stride=16
            stage3: [B, 320, 20, 20],  stride=32
        """
        outs = []

        # stem: stride 4
        x = self.stem(x)

        for idx, stage in enumerate(self.stages):
            x = stage(x)

            if idx in self.return_idx:
                outs.append(x)

            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        return outs


if __name__ == "__main__":
    model = FasterNet(name="t1", return_idx=[0, 1, 2, 3])
    x = torch.randn(2, 3, 640, 640)
    ys = model(x)
    for i, y in enumerate(ys):
        print(i, y.shape)
        
        
        
# 整体模型结构：stem + Stage*4      （每个stage由多个block堆叠而成，数量为[1,2,8,2]）不同stage之间用downsample卷积连接
# 链路中的stem和downsample实现都很简洁，一个卷积层