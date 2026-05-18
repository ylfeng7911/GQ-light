import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init 
import math 
from torchvision.ops import DeformConv2d


from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import get_activation


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            groups = groups,
            bias=bias,
            )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes,device = query.device)    #加了device
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output
    
## BiFPN
class SeparableConv2d(nn.Module):
    """Depthwise + Pointwise Conv"""
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class BiFPN_WeightedFusion(nn.Module):
    """可学习加权融合 (Fast Normalized Fusion)"""
    def __init__(self, num_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        # Normalize the weights
        w = F.relu(self.weights)
        weight = w / (torch.sum(w, dim=0) + 1e-4)
        fused = 0
        for i in range(len(inputs)):
            fused += weight[i] * inputs[i]
        return fused

class BiFPNLayer(nn.Module):
    """单层 BiFPN 融合逻辑"""
    def __init__(self, channels, act=nn.SiLU()):
        """
        channels: list of feature map channels, e.g. [C3, C4, C5]    --input 
        """
        super().__init__()
        self.num_levels = len(channels)         #输入尺度数

        # 横向 1×1 conv 调整通道(可省?)
        self.lateral_convs = nn.ModuleList([
            ConvNormLayer(c, channels[0], 1, 1, act=act) for c in channels
        ])

        # 自顶向下融合 (FPN)
        self.top_down_fusions = nn.ModuleList([
            BiFPN_WeightedFusion(2) for _ in range(self.num_levels - 1)
        ])
        self.top_down_convs = nn.ModuleList([
            SeparableConv2d(channels[0], channels[0], act=act) for _ in range(self.num_levels - 1)
        ])

        # 自底向上融合 (PAN)
        self.bottom_up_fusions = nn.ModuleList([
            BiFPN_WeightedFusion(2) for _ in range(self.num_levels - 1)
        ])
        self.bottom_up_convs = nn.ModuleList([
            SeparableConv2d(channels[0], channels[0], act=act) for _ in range(self.num_levels - 1)
        ])

    def forward(self, inputs):      #inputs是从大到小
        """
        inputs: list of feature maps [P3, P4, P5]
        """
        assert len(inputs) == self.num_levels
        feats = [lateral(x) for x, lateral in zip(inputs, self.lateral_convs)]

        # ----- Top-down path -----     上采样+加权融合+卷积层
        for i in range(self.num_levels - 1, 0, -1):     # 2,1
            up = F.interpolate(feats[i], size=feats[i - 1].shape[-2:], mode='nearest')
            feats[i - 1] = self.top_down_fusions[i - 1]([feats[i - 1], up])
            feats[i - 1] = self.top_down_convs[i - 1](feats[i - 1])

        # ----- Bottom-up path -----    最大值池化+加权融合+卷积层
        for i in range(0, self.num_levels - 1):         # 0,1
            down = F.max_pool2d(feats[i], 2, stride=2)
            feats[i + 1] = self.bottom_up_fusions[i]([feats[i + 1], down])
            feats[i + 1] = self.bottom_up_convs[i](feats[i + 1])

        return feats


class BiFPN(nn.Module):
    """多层堆叠的 BiFPN"""
    def __init__(self, in_channels, out_channels, num_layers=6, act=nn.SiLU()):
        super().__init__()
        self.num_layers = num_layers                    #2
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(in_channels, act=act) for _ in range(num_layers)
        ])
        # 输出层 1x1 conv 保持一致通道
        self.out_convs = nn.ModuleList([
            ConvNormLayer(256, out_channels, 1, 1, act=act)
            for _ in range(len(in_channels))
        ])

    def forward(self, inputs):
        for layer in self.bifpn_layers:
            inputs = layer(inputs)
        outs = [conv(x) for conv, x in zip(self.out_convs, inputs)]     #每个尺度特征图做1x1卷积(或许可省)
        return outs



## VAN-DETR中的HLFFF
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DWConv(nn.Module):
    """
    Depth-wise convolution.
    
    核心思想：每个输入通道由一个独立的卷积核处理。
    关键参数：groups = in_channels
    """
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, d=1, act=None):
        super().__init__()
        # 关键修正：groups必须等于输入通道数c1，实现每个通道独立卷积
        self.dwconv = nn.Conv2d(c1, c1, kernel_size, stride, autopad(kernel_size, padding, d), groups=c1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.bn(self.dwconv(x)))


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution.
    
    核心思想：由 Depth-wise (DW) 和 Point-wise (PW) 两部分组成。
    1. DW卷积：进行空间特征提取，不改变通道数。
    2. PW卷积：使用1x1卷积进行通道间的信息融合和通道数变换。
    """
    def __init__(self, c1, c2, k=3, s=1, d=1, act="RELU"):
        super().__init__()
        # 1. Depth-wise卷积，只负责空间维度（g=c_in）
        self.dw = DWConv(c1, c1, kernel_size=k, stride=s, d=d, act=act)
        
        # 2. Point-wise卷积，1x1卷积
        self.pw = ConvNormLayer(c1, c2, kernel_size=1, stride=1, act=act)

    def forward(self, x):
        # 先进行DW卷积，再进行PW卷积
        return self.pw(self.dw(x))


class EFF(nn.Module):           #输入2同尺度特征图，输出融合特征图
    def __init__(self, in_C, out_C):
        super(EFF, self).__init__()
        self.RGB_K = DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)             #concat再减半c
        self.INF_K = DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)
        self.Second_reduce = DSConv(in_C, out_C, 3) #concat再减半c
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))                
        RGB_K = self.RGB_K(x)                          
        RGB_V = self.RGB_V(x)                           
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)         #[b,256,hw]
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)        #[b,hw,256]
        RGB_Q = Q.view(m_batchsize, -1, width * height)     #[b,256,hw]
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)           #计算注意力得分
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))        #注意力加权
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)    #[b,256,h,w]
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)        #[b,hw,256]
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class EFF2(nn.Module):           #输入2同尺度特征图，输出融合特征图
    def __init__(self, in_C, out_C):
        super(EFF2, self).__init__()
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)             #concat再减半c
        self.INF_V = DSConv(out_C, out_C, 3)
        self.Second_reduce = DSConv(in_C, out_C, 3) #concat再减半c

        self.attn1 = MSDeformableAttention(embed_dim=256, num_heads=8, num_levels=1, num_points=4)
        self.attn2 = MSDeformableAttention(embed_dim=256, num_heads=8, num_levels=1, num_points=4)
    
    @staticmethod
    def get_reference_points(spatial_shapes, bs, device):
        # 每层均按规则网格生成坐标，归一化到 [0,1]。
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 均匀采样网格中心
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij'  
            )
            # 归一化到 [0, 1]
            ref_y = (ref_y.reshape(-1)[None]) / H_
            ref_x = (ref_x.reshape(-1)[None]) / W_
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_*W_, 2]
            reference_points_list.append(ref)
        # 拼接所有尺度
        reference_points = torch.cat(reference_points_list, dim=1)  # [1, sum(H_l*W_l), 2]
        # 扩展到 [bs, sum(H_l*W_l), num_levels, 2]，与 deformable attention 接口对齐
        reference_points = reference_points[:, :, None, :].repeat(bs, 1, len(spatial_shapes), 1)
        return reference_points
    
    
    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))           
        RGB_V = self.RGB_V(x)                           
        m_batchsize, C, height, width = RGB_V.size()
        reference_points = self.get_reference_points([[height,width]],m_batchsize,x.device)
        RGB_V = RGB_V.view(m_batchsize, -1, width * height).permute(0, 2, 1)         #[b,hw,256]
        RGB_Q = Q.view(m_batchsize, -1, width * height).permute(0, 2, 1)      #[b,hw,256]
        RGB_refine = self.attn1(  
            RGB_Q,                              #[b,hw,256]
            reference_points,                   #[b,hw,1,4]
            RGB_V,                              #[b,hw,256]
            [[height,width]])                   #[[h,w]]
        RGB_refine = RGB_refine.permute(0, 2, 1).view(m_batchsize, -1, height, width) 
        RGB_refine = RGB_refine + y

        INF_V = self.RGB_V(y)                           
        reference_points = self.get_reference_points([[height,width]],m_batchsize,x.device)
        INF_V = INF_V.view(m_batchsize, -1, width * height).permute(0, 2, 1)         #[b,hw,256]
        INF_Q = Q.view(m_batchsize, -1, width * height).permute(0, 2, 1)      #[b,hw,256]
        INF_refine = self.attn2(  
            INF_Q,                              #[b,hw,256]
            reference_points,                   #[b,hw,1,4]
            INF_V,                              #[b,hw,256]
            [[height,width]])                   #[[h,w]]
        INF_refine = INF_refine.permute(0, 2, 1).view(m_batchsize, -1, height, width) 
        INF_refine = INF_refine + x
        
        return self.Second_reduce(torch.cat([RGB_refine,INF_refine], dim=1))



class DenseLayer(nn.Module):  #不改变尺寸、通道数
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor           # 256/4

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):      # 2层DSConv，不变hw，
            self.denseblock.append(DSConv(mid_C * i, mid_C, 3))

        self.fuse = DSConv(in_C + mid_C, out_C, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)     #压缩c
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))   #这次的特征 cat 前几次的特征. 对列表的解包（传参用）
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)  #输入特征和融合后特征cat
        return self.fuse(feats)


class HLFFF(nn.Module):         #输入1是最高级特征图（调整通道），输入2是PAN输出的最高级图
    def __init__(self, Channel):
        super(HLFFF, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)      #两路分别DenseLayer，再EFF模块融合
        self.Infobj = DenseLayer(Channel, Channel)
        self.obj_fuse = EFF2(Channel * 2, Channel)

    def forward(self, feat1, feat2):
        rgb_sum = self.RGBobj(feat1)
        Inf_sum = self.Infobj(feat2)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out

    



class ChannelAttention(nn.Module):
    """
    通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(), 
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),                                              #添加正则化
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            # nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       avg_out = self.fc(self.avg_pool(x))      #[b,c,1,1] 
       max_out = self.fc(self.max_pool(x))      #[b,c,1,1] 
       out = avg_out + max_out
       out = self.sigmoid(out)                  #[b,c,1,1]每个通道的权重
       return out * x                           #每个通道按权重缩放[b,c,h,w]


class SpatialAttention(nn.Module):
    """
    空间注意力
    """

    def __init__(self, kernel_size=7, dropout=0.1):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        #添加正则化
        self.norm = nn.BatchNorm2d(1)  # 对每个通道/位置的值做归一化
        self.dropout = nn.Dropout(p=dropout)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)            #[b,1,h,w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)          #[b,1,h,w]
        out = torch.cat([avg_out, max_out], dim=1)              #[b,2,h,w]
        out = self.conv1(out)
        out = self.norm(out)
        out = self.sigmoid(out)                     #[b,1,h,w]
        out = self.dropout(out)  
        return out * x                          #每个空间位置缩放[b,c,h,w]


class CBAM(nn.Module):
    """
    CBAM混合注意力机制.作用：让模型知道特征图的哪些通道、区域是 重要/不重要，从而放大/缩小。
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3, dropout=0.05):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)        #添加正则化
          
    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return self.dropout(x)


#正常CSAF
class CrossScaleAlignFuse(nn.Module):
    def __init__(self, c, deform_groups=4):
        super().__init__()
        self.CBAM_P2 = CBAM(256)
        self.CBAM_P5 = CBAM(256)
        self.deform_align = DeformConv2d(c, c, kernel_size=3, padding=1, groups=deform_groups)
        self.offset_gen = nn.Conv2d(c, 2*3*3*deform_groups, 3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(c*2, c, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(c, c, 1)

    def forward(self, P2, P5):
        P2 = self.CBAM_P2(P2)
        P5 = self.CBAM_P5(P5)
        # 上采样P5到P2尺度
        P5_up = F.interpolate(P5, size=P2.shape[-2:], mode='bilinear', align_corners=False)
        offset = self.offset_gen(P2)        #希望P5对齐到P2
        P5_aligned = self.deform_align(P5_up, offset)

        # 门控融合
        g = self.gate(torch.cat([P2, P5_aligned], dim=1))
        fused = P2 * g + P5_aligned * (1 - g)
        # 门控 改为 残差融合
        # beta = 0.6
        # fused = P2 + beta * (P5_aligned - P2)
        return self.out_proj(fused)
    
#只有P2
# class CrossScaleAlignFuse(nn.Module):
#     def __init__(self, c, deform_groups=4):
#         super().__init__()
#         self.CBAM_P2 = CBAM(256)
#         self.out_proj = nn.Conv2d(c, c, 1)

#     def forward(self, P2, P5):
#         P2 = self.CBAM_P2(P2)
#         return self.out_proj(P2)
    
#只有p5  
# class CrossScaleAlignFuse(nn.Module):
#     def __init__(self, c, deform_groups=4):
#         super().__init__()
#         self.CBAM_P5 = CBAM(256)
#         self.out_proj = nn.Conv2d(c, c, 1)

#     def forward(self, P2, P5):
#         P5 = self.CBAM_P5(P5)
#         return self.out_proj(P5)
    





class DeformAlignONNX(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.offset_gen = nn.Conv2d(c, 2, 3, 1, 1)  # 简化：per-pixel offset
        self.proj = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset_gen(x)          # [B, 2, H, W]

        # build base grid (static, cacheable)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        base_grid = base_grid.unsqueeze(0).repeat(B,1,1,1)

        # normalize offset
        offset = offset.permute(0,2,3,1)      # [B,H,W,2]
        offset = offset / torch.tensor([W, H], device=x.device)

        grid = base_grid + offset
        x_aligned = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        return self.proj(x_aligned)


class CrossScaleAlignFuseONNX(nn.Module):
    def __init__(self, c, deform_groups=4):
        super().__init__()
        self.CBAM_P2 = CBAM(256)
        self.CBAM_P5 = CBAM(256)
        self.deform_align = DeformAlignONNX(c)
        self.gate = nn.Sequential(
            nn.Conv2d(c*2, c, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(c, c, 1)

    def forward(self, P2, P5):
        P2 = self.CBAM_P2(P2)
        P5 = self.CBAM_P5(P5)
        # 上采样P5到P2尺度
        P5_up = F.interpolate(P5, size=P2.shape[-2:], mode='bilinear', align_corners=False)
        P5_aligned = self.deform_align(P5_up)

        # 门控融合
        g = self.gate(torch.cat([P2, P5_aligned], dim=1))
        fused = P2 * g + P5_aligned * (1 - g)
        return self.out_proj(fused)





# --- 计算参数量的函数 ---
def count_parameters(model):
    """计算模型的总参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def print_model_parameter_details(model):
    """打印模型每个子模块的参数量"""
    total_params = 0
    print(f"{'模块名称':<30} {'参数量':<15}")
    print("-" * 45)
    for name, module in model.named_modules():
        # 只打印叶子模块（即没有子模块的模块）的参数，避免重复计算
        if not list(module.children()):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:<30} {params:<15,}")
                total_params += params
    print("-" * 45)
    print(f"{'总参数量':<30} {total_params:<15,}")
    return total_params

if __name__ == '__main__':
    # 1. 定义参数
    batch_size = 4
    channels = 256  # in_C = out_C = 256
    height, width = 20, 20

    # 2. 创建随机输入数据
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    input_data = (x, y)

    print(f"输入 x 的形状: {x.shape}")
    print(f"输入 y 的形状: {y.shape}")
    print("-" * 30)

    # 3. 实例化 HLFFF 模块
    hlfff_module = HLFFF(Channel=channels)

    # 4. 前向传播
    output_tensor = hlfff_module(input_data)
    print(f"✅ HLFFF 模块运行成功！")
    print(f"输出张量的形状: {output_tensor.shape}")
    # 预期输出形状: [4, 256, 20, 20]
    assert output_tensor.shape == (batch_size, channels, height, width)
    print("✅ 输出形状符合预期！")
    
    print(f"正在计算 HLFFF (Channel={channels}) 模块的参数量...\n")

    # 3. 打印详细的参数量分析
    total_params = print_model_parameter_details(hlfff_module)
    
    # 4. (可选) 验证总参数量计算是否正确
    print(count_parameters(hlfff_module))
    assert total_params == count_parameters(hlfff_module)
    print("\n✅ 参数量计算完成并验证无误！")
