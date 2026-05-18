from typing import Callable, List, Optional
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ...core import register

# =========================================================
# 工具函数：将通道数调整为 8 的倍数
# 目的：更适合移动端 / TensorRT / 硬件加速器
# =========================================================
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# =========================================================
# 基础模块：Conv + BN + Activation
# 可用于普通卷积、1x1卷积、DW卷积（通过 groups=in_channels 实现）
# =========================================================
class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        padding = (kernel_size - 1) // 2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )


# =========================================================
# SE 模块: 在不同通道上乘一个权重
# =========================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


# =========================================================
# MobileNetV3 的 block 配置类
# 每个 block 指定：输入通道、卷积核大小、扩张通道、输出通道、是否SE、激活类型、步长
# =========================================================
class InvertedResidualConfig:
    def __init__(
        self,
        input_c: int,
        kernel: int,
        expanded_c: int,
        out_c: int,
        use_se: bool,
        activation: str,
        stride: int,
        width_multi: float,
    ):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


# =========================================================
# MobileNetV3 的基本倒残差块
#
# 结构：
#   1) 可选 1x1 expand
#   2) kxk depthwise conv
#   3) 可选 SE
#   4) 1x1 project
#   5) 满足条件时 residual shortcut
# =========================================================
class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)       #需要stride=1并且c不变，才会加跳跃连接
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        layers: List[nn.Module] = []

        # 1) expand 1x1 conv（某些block第一步不扩张）
        if cnf.expanded_c != cnf.input_c:   #扩张c
            layers.append(
                ConvBNActivation(
                    cnf.input_c,
                    cnf.expanded_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # 2) depthwise conv         3x3或5x5，stride=1或2
        # 逐深度卷积，组数 = c_input
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,      #组数等于通道数，即一个卷积核对应一个通道，通道之间不混合
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # 3) optional SE   有些用有些不用
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # 4) project 1x1 conv（注意这里不加非线性）
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.stride = cnf.stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out


# =========================================================
# MobileNetV3-Large 检测骨干版
#
# 输入:
#   x: [B, 3, H, W]
#
# 输出:
#   outs = [C2, C3, C4, C5]
#   按尺度从大到小排列，即:
#       C2: stride = 4
#       C3: stride = 8
#       C4: stride = 16
#       C5: stride = 32
#
# 注意：
# 1) 这里只保留 backbone，不包含分类头
# 2) 返回的是多尺度特征图列表，适合接到 RT-DETR / FPN / PAN / 自定义neck
# =========================================================
@register()
class MobileNetV3LargeBackbone(nn.Module):
    def __init__(
        self,
        width_multi: float = 1.0,
        reduced_tail: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        return_idx: List[int] = [0, 1, 2, 3],  # 返回哪些尺度，默认全返回
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
        reduce_divider = 2 if reduced_tail else 1       # 1

        # -------------------------------------------------
        # MobileNetV3-Large 原始 block 配置
        # 每一项:
        # (input_c, kernel, expanded_c, out_c, use_se, act, stride)
        # -------------------------------------------------
        self.block_configs = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2),   # stride 4 开始
            bneck_conf(24, 3, 72, 24, False, "RE", 1),

            bneck_conf(24, 5, 72, 40, True, "RE", 2),    # stride 8 开始
            bneck_conf(40, 5, 120, 40, True, "RE", 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1),

            bneck_conf(40, 3, 240, 80, False, "HS", 2),  # stride 16 开始
            bneck_conf(80, 3, 200, 80, False, "HS", 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1),

            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # stride 32 开始
            bneck_conf(160 // reduce_divider,5,960 // reduce_divider,160 // reduce_divider,True,"HS",1,),
            bneck_conf(160 // reduce_divider,5,960 // reduce_divider,160 // reduce_divider,True,"HS",1,),
        ]

        # -------------------------------------------------
        # stem
        # 输入:  [B, 3, H, W]
        # 输出:  [B, 16, H/2, W/2]
        # -------------------------------------------------
        firstconv_output_c = self.block_configs[0].input_c
        self.stem = ConvBNActivation(
            3,
            firstconv_output_c,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        # -------------------------------------------------
        # 所有 inverted residual blocks
        # -------------------------------------------------
        self.blocks = nn.ModuleList([InvertedResidual(cnf, norm_layer) for cnf in self.block_configs])

        # -------------------------------------------------
        # 最后的 1x1 conv
        # 对应 backbone 最深层特征再增强一次
        #
        # 输入:  [B, last_out_c, H/32, W/32]
        # 输出:  [B, 6 * last_out_c, H/32, W/32]
        # -------------------------------------------------
        lastconv_input_c = self.block_configs[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        self.last_conv = ConvBNActivation(
            lastconv_input_c,
            lastconv_output_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        # -------------------------------------------------
        # 我们定义4个输出尺度：
        #
        # C2: stride=4
        # C3: stride=8
        # C4: stride=16
        # C5: stride=32
        #
        # 注意这里是“按空间尺度变化节点”划分stage，而不是硬套ResNet命名。
        # -------------------------------------------------
        #
        # stem 之后是 stride=2
        # block 1 后仍然 stride=2
        # block 2 后第一次到 stride=4  -> C2
        # block 4 后第一次到 stride=8  -> C3
        # block 7 后第一次到 stride=16 -> C4
        # block 13 后第一次到 stride=32 -> C5_pre
        # 再经过 last_conv 得到最终 C5
        #
        self.return_idx = return_idx
        self.out_channels = [
            self.block_configs[2].out_c,   # C2 = 24
            self.block_configs[5].out_c,   # C3 = 40
            self.block_configs[11].out_c,  # C4 = 112
            lastconv_output_c,             # C5 = 960 (width=1.0, reduced_tail=False)
        ]
        self.out_strides = [4, 8, 16, 32]

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        输入:
            x: [B, 3, H, W]

        输出:
            outs: list[Tensor]
                默认包含4个尺度, 按尺度从大到小:
                - outs[0]: stride=4
                - outs[1]: stride=8
                - outs[2]: stride=16
                - outs[3]: stride=32
        """
        outs = []

        # stem
        # x: [B, 3, H, W] -> [B, 16, H/2, W/2]
        x = self.stem(x)

        # 逐个 block 前向，并在指定位置截取特征
        #
        # block索引与尺度对应关系：
        # block[0]  -> stride=2
        # block[1]  -> stride=4
        # block[2]  -> stride=4     => C2
        # block[3]  -> stride=8
        # block[4]  -> stride=8
        # block[5]  -> stride=8     => C3
        # block[6]  -> stride=16
        # ...
        # block[11] -> stride=16    => C4
        # block[12] -> stride=32
        # block[13] -> stride=32
        # block[14] -> stride=32
        #
        for i, block in enumerate(self.blocks):
            x = block(x)

            # C2: stride = 4, shape ~= [B, 24, H/4, W/4]
            if i == 2:
                c2 = x

            # C3: stride = 8, shape ~= [B, 40, H/8, W/8]
            elif i == 5:
                c3 = x

            # C4: stride = 16, shape ~= [B, 112, H/16, W/16]
            elif i == 11:
                c4 = x

            # C5_pre: stride = 32, shape ~= [B, 160, H/32, W/32]
            elif i == 14:
                c5 = self.last_conv(x)  # [B, 960, H/32, W/32]

        all_feats = [c2, c3, c4, c5]

        # 根据 return_idx 选择返回哪些尺度
        outs = [all_feats[i] for i in self.return_idx]
        return outs


# =========================================================
# 工厂函数
# 方便你像 PResNet 一样实例化
# =========================================================
def mobilenet_v3_large_backbone(
    width_multi: float = 1.0,
    reduced_tail: bool = False,
    return_idx: List[int] = [0, 1, 2, 3],
) -> MobileNetV3LargeBackbone:
    return MobileNetV3LargeBackbone(
        width_multi=width_multi,
        reduced_tail=reduced_tail,
        return_idx=return_idx,
    )


# =========================================================
# 简单测试
# =========================================================
if __name__ == "__main__":
    model = mobilenet_v3_large_backbone()
    x = torch.randn(2, 3, 704,704)
    feats = model(x)

    print("Number of output feature maps:", len(feats))
    for i, f in enumerate(feats):
        print(
            f"outs[{i}]: shape={tuple(f.shape)}, "
            f"stride={model.out_strides[i]}, channels={f.shape[1]}"
        )

    # 对于 640x640 输入，预期大致是：
    # outs[0]: [2, 24, 160, 160]   stride=4
    # outs[1]: [2, 40, 80, 80]     stride=8
    # outs[2]: [2, 112, 40, 40]    stride=16
    # outs[3]: [2, 960, 20, 20]    stride=32
    # 输出通道数为：[24,40,112,960]