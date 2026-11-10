import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvNormLayer


def _build_dct_matrix(n: int, device, dtype):
    """
    构造正交归一化 DCT-II 变换矩阵 C。

    对输入 x:
        DCT(x)  = C @ x
        IDCT(X) = C.T @ X

    2D 情况:
        DCT2(x)  = C_h @ x @ C_w.T
        IDCT2(X) = C_h.T @ X @ C_w
    """
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, n)

    mat = torch.cos(math.pi / n * (i + 0.5) * k)

    mat[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= math.sqrt(2.0 / n)

    return mat


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D DCT-II, norm='ortho'
    输入:
        x: [B, C, H, W]
    输出:
        y: [B, C, H, W]
    """
    b, c, h, w = x.shape
    dtype = x.dtype
    device = x.device

    # DCT 矩阵建议用 float32 计算，AMP 下更稳定
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    x_ = x.to(compute_dtype)
    c_h = _build_dct_matrix(h, device, compute_dtype)
    c_w = _build_dct_matrix(w, device, compute_dtype)

    x_ = x_.reshape(b * c, h, w)
    y = torch.matmul(c_h, x_)
    y = torch.matmul(y, c_w.t())
    y = y.reshape(b, c, h, w)

    return y.to(dtype)


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D IDCT, norm='ortho'
    输入:
        x: [B, C, H, W]
    输出:
        y: [B, C, H, W]
    """
    b, c, h, w = x.shape
    dtype = x.dtype
    device = x.device

    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    x_ = x.to(compute_dtype)
    c_h = _build_dct_matrix(h, device, compute_dtype)
    c_w = _build_dct_matrix(w, device, compute_dtype)

    x_ = x_.reshape(b * c, h, w)
    y = torch.matmul(c_h.t(), x_)
    y = torch.matmul(y, c_w)
    y = y.reshape(b, c, h, w)

    return y.to(dtype)


class DctSpatialInteraction(nn.Module):
    """
    HFP 的 Spatial Path。

    对于浅层特征，例如 P1/P2 或 C2/C3:
        isdct=True，使用 DCT 高通滤波生成高频响应，再作为空间增强 mask。

    对于高层特征，例如 P3/P4 或 C4/C5:
        isdct=False，不做 DCT，而是使用 1x1 conv 生成空间权重。
    """
    def __init__(self, in_channels, ratio=(0.25, 0.25), isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct

        if not self.isdct:
            self.spatial1x1 = ConvNormLayer(
                in_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act='relu',
            )

    def forward(self, x):
        _, _, h, w = x.size()

        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        # DCT 高通滤波
        x_dct = dct_2d(x)       #此时后2维度从[h,w]意义变为了[u,v]频率位置，小u小v表示低频，大u大v表示高频
        weight = self._compute_weight(h, w, self.ratio, x.device, x.dtype)
        weight = weight.view(1, 1, h, w)
        x_high = x_dct * weight         #低频部分过滤掉
        spatial_mask = idct_2d(x_high)          # 逆变换回来，重新成为特征图（滤除低频部分）
        spatial_mask = torch.sigmoid(torch.abs(spatial_mask))


        # 注意：原代码这里没有 sigmoid，直接 x * dct_      todo：尝试+sigmoid有无改进
        return x * spatial_mask

    @staticmethod
    def _compute_weight(h, w, ratio, device, dtype):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])

        weight = torch.ones((h, w), device=device, dtype=dtype)
        weight[:h0, :w0] = 0

        return weight


class DctChannelInteraction(nn.Module):
    """
    HFP 的 Channel Path。

    DCT 分支:
        先用 DCT 高通滤波得到高频响应；
        再分别做 adaptive max pooling 和 adaptive avg pooling 到 patch 大小；
        ReLU 后在空间维度求和；
        使用 group 1x1 conv 生成 channel attention。

    非 DCT 分支:
        直接对原特征做 GAP/GMP 到 1x1，然后生成 channel attention。
    """
    def __init__(
        self,
        in_channels,
        patch=(8, 8),
        ratio=(0.25, 0.25),
        isdct=True,
        groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct

        assert in_channels % groups == 0, \
            f"in_channels={in_channels} must be divisible by groups={groups}"

        self.channel1x1 = ConvNormLayer(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
            act='relu',
        )

        self.channel2x1 = ConvNormLayer(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
            act='relu',
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, h, w = x.size()

        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))

            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))

            return x * torch.sigmoid(self.channel2x1(channel))

        # DCT 高频滤波
        x_dct = dct_2d(x)
        weight = self._compute_weight(h, w, self.ratio, x.device, x.dtype)
        weight = weight.view(1, 1, h, w)
        x_high = x_dct * weight
        x_high = idct_2d(x_high)            #得到滤除低频信息的特征图

        # 原代码中不是直接 GAP/GMP 到 1x1，而是先池化到 patch，再 sum
        amaxp = F.adaptive_max_pool2d(x_high, output_size=(self.h, self.w))     #自动划分为8x8个区域，每个区域做最大值池化
        aavgp = F.adaptive_avg_pool2d(x_high, output_size=(self.h, self.w))

        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)           #amaxp很大，aavgp很小，不太均衡

        return x * torch.sigmoid(self.channel2x1(channel))

    @staticmethod
    def _compute_weight(h, w, ratio, device, dtype):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])

        weight = torch.ones((h, w), device=device, dtype=dtype)
        weight[:h0, :w0] = 0

        return weight


# 输入（大尺度）特征图，用其中的高频（小目标）信息引导特征融合。输出h,w,c不变。
class HFP(nn.Module):
    """
    High Frequency Perception Module.

    Args:
        in_channels: 输入通道数，RT-DETR/GQ-DETR 中通常是 hidden_dim=256。
        ratio: DCT 高通滤波比例，例如论文中 alpha=0.25，可写成 (0.25, 0.25)。
        patch: Channel Path 中 adaptive pooling 的输出尺寸。
               原论文描述 k 通常设置为 16，但你给的源码默认是 (8, 8)，这里遵照源码。
        isdct: 是否启用 DCT 高频滤波。
               对浅层大分辨率特征建议 True；
               对高层小分辨率特征可设 False。
        groups: Channel Path 中 group 1x1 conv 的 groups，源码是 32。
    """
    def __init__(
        self,
        in_channels,
        ratio=(0.25, 0.25),
        patch=(8, 8),
        isdct=True,
        groups=32,
    ):
        super().__init__()

        self.spatial = DctSpatialInteraction(
            in_channels=in_channels,
            ratio=ratio,
            isdct=isdct,
        )

        self.channel = DctChannelInteraction(
            in_channels=in_channels,
            patch=patch,
            ratio=ratio,
            isdct=isdct,
            groups=groups,
        )

        self.out = nn.Sequential(
            ConvNormLayer(in_channels, in_channels, 3, 1, 1, act = 'relu', groups=in_channels),
            ConvNormLayer(in_channels, in_channels, 1, 1, 0, act = 'relu'),
            nn.GroupNorm(32, in_channels),
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)

        return self.out(spatial + channel)
        # return self.out(spatial)
    

if __name__ == '__main__':
    a = torch.randn(2,256,80,80)
    mo = HFP(256)
    print(mo(a).shape)
    
    def count_params(model, verbose=True):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if verbose:
            print(f"Total params    : {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Total params    : {total_params / 1e6:.4f} M")
            print(f"Trainable params: {trainable_params / 1e6:.4f} M")

        return total_params, trainable_params

    count_params(mo)
    
    
    #参数0.6M，flops 8G
    #修改后：0.07M, 2.0G