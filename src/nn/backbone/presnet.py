"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor

from collections import OrderedDict

from .common import get_activation, FrozenBatchNorm2d

from ...core import register
from ...zoo.rtdetr.modules import MultiKernelPerception, FCM, FCM2

__all__ = ['PResNet']


class PartialConv(nn.Module):
    def __init__(
        self,
        dim: int,
        n_div: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()

        self.dim_conv3 = dim // n_div                   # 只有1/4通道参与卷积（但这是全参数的卷积）
        self.dim_untouched = dim - self.dim_conv3

        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.forward = self.forward_split_cat

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
    

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias,
            groups=groups)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    # stride：只有2,3,4Stage的第一个bb=2，其余为1；
    # shortcut：第二个bb开始做
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', fcm1 = True):   
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:      #如果stride=2就用平均池化下采样
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:                                   #如果stride != 2 就不用池化
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)        

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        # 以下改进效果不是很好。
        # self.branch2a = nn.Sequential(ConvNormLayer(ch_in, ch_in, 3, stride, act=act, groups=ch_in),
        #                                ConvNormLayer(ch_in, ch_out, 1, 1, act=act))
        # self.branch2b = FCM(ch_out, ch_out) if fcm1 else FCM2(ch_out, ch_out)
        
        # 改小K + faster+MB风格改进
        # self.branch2a = nn.Sequential(ConvNormLayer(ch_in, ch_in, 3, stride, act=act, groups=ch_in),
        #                                ConvNormLayer(ch_in, ch_out, 1, 1, act=act))
        # self.branch2b = nn.Sequential(PartialConv(ch_out),
        #                               ConvNormLayer(ch_out, ch_out, 1, 1, act=act))
        
        
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):        #3个卷积 + 残差，  通道数x4
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:       # d
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)     #输出通道数=ch_outx4

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):    #多个bottleNeck的堆叠
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b', fcm1 = True):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):      #深度分别是3,4,6,3
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,       #在2,3,4 Stage的第一个bb做下采样
                    shortcut=False if i == 0 else True,                 #第二个bb开始做跳跃连接
                    variant=variant,
                    act=act,
                    fcm1=fcm1)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register()
class PResNet(nn.Module):
    def __init__(
        self, 
        depth,                  #50
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1,           #0
        freeze_norm=True, 
        pretrained=False):      #True
        super().__init__()

        block_nums = ResNet_cfg[depth]  #[2，2，2，2]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def   #conv1:conv1_1, 1_2, 1_3
        ]))
        
        # MB风格
        # self.conv1 = nn.Sequential(ConvNormLayer(3, ch_in // 2, 3, 2, act=act),
        #                            ConvNormLayer(ch_in // 2, ch_in // 2, 3, 1, act=act, groups = ch_in // 2),
        #                            ConvNormLayer(ch_in // 2, ch_in, 1, 1, act=act),
        #                            )
        # fasternet风格
        # self.conv1 = nn.Sequential(ConvNormLayer(3, ch_in//2, 3, 2, act=act),
        #                             PartialConv(ch_in//2),
        #                             ConvNormLayer(ch_in//2, ch_in, 1, 1, act=act),                   
        #                             )
    

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]      # expansion=1   [256,512,1024,2048]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()   # 构建4个stage子模块
        for i in range(num_stages):     
            stage_num = i + 2           # 2,3,4,5
            fcm1 = True if i % 2==0 else False      #T, F, T, F 
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant, fcm1 = fcm1)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]            # [256,512,1024,2048]
        self.out_strides = [_out_strides[_i] for _i in return_idx]              # [4, 8, 16, 32]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)       #通道3 ---> 64, 尺度/2
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1) #再次尺度/2
        outs = []
        for idx, stage in enumerate(self.res_layers):       #stage是个Blocks模块，包含多个block
            x = stage(x)
            if idx in self.return_idx:  #[0,1,2,3]
                outs.append(x)
        return outs     #3尺度,从大到小

#流程 conv1 ---> max_pool ---> 4个残差stage（每个stage一张特征图）
#每个stage是多个BottleNeck(或Basiclock)的堆叠（在第一个BottleNeck处下采样）
#输出通道数分别是256，512,1024,2048
#下采样率分别为4,8,16,32