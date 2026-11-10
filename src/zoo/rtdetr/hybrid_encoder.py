"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register

from .modules import HLFFF,ConvNormLayer,DSConv, FCM, MultiKernelPerception


__all__ = ['HybridEncoder']



def _save_feat_token_heatmaps(
    feats,
    sample,
    stage_name,
    save_dir="./feature_token_vis",
    names=None,
    batch_idx=0,
    step=None,
    alpha_img=0.55,
    alpha_heat=0.45,
):
    """
    将多尺度特征图的 token 响应以 heatmap 形式叠加到原图灰度图上保存。

    Args:
        feats: list[Tensor]
            多尺度特征图列表，每个元素 shape 为 [B, C, H, W]。
            例如：
                FPN前:  [C2, C3, C4, C5]
                FPN后:  [P2, P3, P4, P5]
                PAN后:  [N2, N3, N4, N5]

        sample: Tensor or NestedTensor
            原图输入，通常 shape 为 [B, C, H, W]。
            如果是 RT-DETR / DETR 中的 NestedTensor，也兼容 sample.tensors。

        stage_name: str
            当前阶段名称，例如：
                "before_fpn"
                "after_fpn"
                "after_pan"
                "after_asff"

        save_dir: str
            保存目录。

        names: list[str]
            每个尺度的名字，例如 ["C2", "C3", "C4", "C5"]。

        batch_idx: int
            可视化 batch 中第几个样本。

        step: int or str or None
            可选，用于区分不同 iteration / epoch。

        alpha_img: float
            原图灰度图叠加权重。

        alpha_heat: float
            heatmap 叠加权重。
    """
    import os
    import cv2
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    if names is None:
        names = [f"L{i}" for i in range(len(feats))]

    def _to_uint8_image(x, b=0):
        """
        将原图 tensor 转为 uint8 RGB 图像。

        输入:
            x: Tensor [B, C, H, W] 或 NestedTensor，支持 x.tensors
        输出:
            img_rgb: np.ndarray [H, W, 3], uint8, RGB 格式
        """
        img = x[b]

        img = img.detach().float().cpu()

        # [1, H, W] -> [3, H, W]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # [C, H, W] -> [H, W, C]
        img = img.permute(1, 2, 0).numpy()

        # min-max normalize 到 [0, 255]
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = (img * 255).astype(np.uint8)

        return img

    def _normalize_heatmap(heatmap):
        heatmap = heatmap.astype(np.float32)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-6)
        return heatmap

    with torch.no_grad():
        # -----------------------------
        # 1. 处理原图
        # -----------------------------
        img_rgb = _to_uint8_image(sample, batch_idx)      # [H, W, 3], RGB
        img_h, img_w = img_rgb.shape[:2]

        # 转为灰度图，再转回 3 通道，便于和 heatmap 叠加
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)          # [H, W]
        img_gray_3c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)      # [H, W, 3], BGR

        # 保存原图灰度图
        if step is None:
            gray_name = f"{stage_name}_input_gray_b{batch_idx}.png"
        else:
            gray_name = f"step{step}_{stage_name}_input_gray_b{batch_idx}.png"

        cv2.imwrite(os.path.join(save_dir, gray_name), img_gray)

        # -----------------------------
        # 2. 逐尺度生成 token heatmap 并叠加到原图上
        # -----------------------------
        for i, feat in enumerate(feats):
            cur = feat.detach().float()

            # cur: [B, C, H, W] -> [C, H, W]
            cur = cur[batch_idx]

            # 每个空间位置视为一个 token
            # 对通道维聚合，得到 token 响应强度图 [H, W]
            heatmap = cur.abs().mean(dim=0)       # [H, W]        =====================================
            # heatmap = F.relu(cur).mean(dim=0)

            heatmap = heatmap.cpu().numpy()
            heatmap = _normalize_heatmap(heatmap)

            # [H_feat, W_feat] -> [H_img, W_img]
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_u8 = cv2.resize(heatmap_u8, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            # cv2 构造彩色 heatmap，输出为 BGR
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

            # heatmap 叠加到原图灰度图上
            overlay = cv2.addWeighted(img_gray_3c, alpha_img, heatmap_color, alpha_heat, 0.0)

            if step is None:
                heat_name = f"{stage_name}_{names[i]}_heat_b{batch_idx}.png"
                overlay_name = f"{stage_name}_{names[i]}_overlay_b{batch_idx}.png"
            else:
                heat_name = f"step{step}_{stage_name}_{names[i]}_heat_b{batch_idx}.png"
                overlay_name = f"step{step}_{stage_name}_{names[i]}_overlay_b{batch_idx}.png"

            # 保存单独 heatmap
            cv2.imwrite(
                os.path.join(save_dir, heat_name),
                heatmap_u8,
            )

            # 保存叠加图
            cv2.imwrite(
                os.path.join(save_dir, overlay_name),
                overlay,
            )
            
            
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


class SKConv(nn.Module):
    def __init__(self, features=256, M=3, G=8, r=2, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU()
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)    #[b,3,c,h,w]
        fea_U = torch.sum(feas, dim=1)          #对x多次卷积，结果相加 [b,c,h,w]
        fea_s = fea_U.mean(-1).mean(-1)         #空间维度求平均 [b,c]
        fea_z = self.fc(fea_s)                  #线性层[b,d]
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)        #[b,1,c]
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)   #[b,3,c]
        attention_vectors = self.softmax(attention_vectors)   #  每个[b,c]都生成一个3维度权重    [b,3,c]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)       #[b,3,c,1,1]
        fea_v = (feas * attention_vectors).sum(dim=1)   #把卷积结果按权重求和
        return fea_v
    
class SK_block(nn.Module):
    def __init__(self, act=None, groups=1):
        super().__init__()
        self.conv = SKConv()
        self.norm = nn.BatchNorm2d(256)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))




class ASFF3(nn.Module):      # 输入：3个不同尺度的特征图，输出：选定尺度的融合特征图    
    def __init__(self, level, rfb=False, vis=False, groups = 8):        # 初始化level:0最高，2最低
        super(ASFF3, self).__init__()
        self.level = level
        self.dim = [256, 256, 256]
        self.inter_dim = self.dim[self.level]   # 256

        if level == 0:
            self.stride_level_1 = ConvNormLayer(256, self.inter_dim, 3, 2, act='silu', groups=groups)
            self.stride_level_2 = ConvNormLayer(256, self.inter_dim, 3, 2, act='silu', groups=groups)
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='silu', groups=groups)

        elif level == 1:
            self.stride_level_2 = ConvNormLayer(256, self.inter_dim, 3, 2, act='silu', groups=groups)
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='silu', groups=groups)

        elif level == 2:
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='silu', groups=groups)

        # 压缩维度
        compress_c = 8 if rfb else 16

        self.weight_level_0 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='silu')
        self.weight_level_1 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='silu')
        self.weight_level_2 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='silu')

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x_level_0, x_level_1, x_level_2):  # 4个尺度特征图   ,输入顺序是从小到大
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(F.max_pool2d(x_level_2, 3, stride=2, padding=1))

        elif self.level == 1:
            level_0_resized = F.interpolate(x_level_0, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = F.interpolate(x_level_0, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # 权重计算
        level_0_weight_v = self.weight_level_0(level_0_resized)     #[b,8,h,w]
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)    #[b,24,h,w]
        levels_weight = self.weight_levels(levels_weight_v)         #[b,3,h,w]
        levels_weight = F.softmax(levels_weight, dim=1)             #[b,3,h,w]

        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :] +
            level_1_resized * levels_weight[:, 1:2, :, :] +
            level_2_resized * levels_weight[:, 2:3, :, :]  )

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return self.dropout(out)





class RepVggBlock(nn.Module):       #(卷积1 + 卷积3)-激活     H/W不变，C不变   特征融合的模块
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)        #3x3卷径，捕捉空间+通道特征
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)        #1x1卷积，捕捉通道特征
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):                       #推理时
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)           #训练时

        return self.act(y)

    def convert_to_deploy(self):    #重参数化（把1x1卷积和3x3卷积融合为一个3x3卷积），加快速度
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):       #H/W不变   特征融合模块
    def __init__(self,
                 in_channels,               #512
                 out_channels,              #256
                 num_blocks=3,      
                 expansion=1.0,             #r18时是0.5
                 bias=None,
                 act="silu",
                 ifFCM = False):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        
        # 特征融合模块  
        if ifFCM:
            self.bottlenecks = nn.Sequential(*[FCM(hidden_channels, hidden_channels) for _ in range(num_blocks)])   #特征融合模块
        else:
            self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)         #就是3个RepVgg特征融合模块串联
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,             # r18使用0.5
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2',
                 ifEncoder = True,
                 ifFCM = False):     # v1
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        self.ifEncoder = ifEncoder
        
        self.save_token_heatmap = False
        self.token_heatmap_dir = "./feature_token_vis"
        self.token_heatmap_step = "vis"
        
        
        # channel projection    编码器入口的映射是为了调整特征图的通道数，这里max图本来就256的，可以不用映射
        self.input_proj = nn.ModuleList()

        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
                       
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        if ifEncoder:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim, 
                nhead=nhead,
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                activation=enc_act)

            self.encoder = nn.ModuleList([
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
            ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):        #3,2
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act)) #1卷积-归一化-激活函数     C/H/W不变
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, ifFCM=ifFCM))#降通道的特征融合  C减半，H/W不变

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):       #0,1
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))  #C不变，H/W减半
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, ifFCM=ifFCM)        #C减半,H/W不变
            )
            

        
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, samples=None):
        assert len(feats) == len(self.in_channels)
        #[128, 256, 512]
        # input_proj前特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                feats,
                samples, 
                stage_name="before_input_proj",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["C3", "C4", "C5"],
                step=getattr(self, "token_heatmap_step", None),
            )
        
        
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]     #1x1卷积--> c = 256  顺序从下往上
        
        # FPN前特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                proj_feats,
                samples, 
                stage_name="before_fpn",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["C3", "C4", "C5"],
                step=getattr(self, "token_heatmap_step", None),
            )
            
        # encoder   最高级特征图 应用编码器（交互语义信息）
        if self.ifEncoder:
            if self.num_encoder_layers > 0:
                for i, enc_ind in enumerate(self.use_encoder_idx):  #0,2
                    h, w = proj_feats[enc_ind].shape[2:]
                    # flatten [B, C, H, W] to [B, HxW, C]
                    src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1) #展平
                    if self.training or self.eval_spatial_size is None:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                    else:
                        pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)       #

                    memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                    proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()   #[b,c,h,w]

            # encoder后特征可视化
            if getattr(self, "save_token_heatmap", False):
                _save_feat_token_heatmaps(
                    [proj_feats[-1]],
                    samples, 
                    stage_name="after_encoder",
                    save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                    names=["E5"],
                    step=getattr(self, "token_heatmap_step", None),
                )
        

        # broadcasting and fusion
        # FPN：inner_outs变为：从下往上3层融合特征图(代码逻辑很绕，结合论文图来看)，FPN网络是自顶向下的，融合方式是concat+映射
        inner_outs = [proj_feats[-1]]      
        for idx in range(len(self.in_channels) - 1, 0, -1):     #2,1        理解：上层（论文中F5和S4）经1x1卷积+2倍插值和下层融合
            feat_heigh = inner_outs[0]         
            feat_low = proj_feats[idx - 1]      
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)    #1x1卷积，下标是0,1,2
            inner_outs[0] = feat_heigh         
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')  #上采样
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))#concat再减半C
            inner_outs.insert(0, inner_out)        

        
        # FPN后特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                inner_outs,
                samples, 
                stage_name="after_fpn",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["P3", "P4", "P5"],
                step=getattr(self, "token_heatmap_step", None),
            )
        
        
        # PAN：outs是：E1和每一次PAN融合后的结果（尺度从下往上)      [b,256,h,w]
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):        #0,1           理解：下层（论文中）经过下采样后和上层融合
            feat_low = outs[-1]         # E1 第一次PAN融合的结果 
            feat_height = inner_outs[idx + 1]       #E2  E3
            downsample_feat = self.downsample_convs[idx](feat_low)      #下采样E3
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1)) #concat再减半C
            outs.append(out)

        
        # # PAN后特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                outs,
                samples, 
                stage_name="after_pan",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["N3", "N4", "N5"],
                step=getattr(self, "token_heatmap_step", None),
            )
            

        return outs            #尺度从大到小

