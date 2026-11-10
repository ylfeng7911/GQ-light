"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch.nn.init as init 
import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import os
import numpy as np
import cv2

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .group_attention import grouped_self_conv,grouped_self_con_one_stage,GatedResidualBlock1d
from .modules import MSDeformableAttention,CrossScaleAlignFuse,CrossScaleAlignFuseONNX
from ...core import register

from .HFP import HFP


__all__ = ['RTDETRTransformer']


def visualize_groups_on_image(
    inter_ref_bbox, # [b, 300, 4], cxcywh, 归一化坐标
    labels,         # [ 300], 一阶段分组标签
    samples,        # [b, 3, h, w], 增强后的图像张量
    save_dir, 
    layer_idx,
    new_labels,     # [300], 二阶段分组标签
    fg_cluster_idx,
    targets,
    logits,
):
    """
    在增强后的图像上绘制不同解码器层的预测框。
    可视化一阶段和二阶段的分组情况，并分别保存。
    """
    # --- 辅助函数：执行核心的可视化和保存逻辑 ---
    def _draw_and_save(labels_to_use, suffix):
        """
        根据给定的标签进行绘制，并用指定的后缀保存文件。
        """
        # 1. 获取增强后图像的尺寸和numpy数组
        _, _, img_h, img_w = samples.shape
        img_tensor = samples[0]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # 2. 处理预测框和标签
        boxes_cxcywh_norm = inter_ref_bbox[0].detach().cpu().numpy()
        labels_np = labels_to_use.detach().cpu().numpy()
        
        # logits形状: [1, 300, 2] -> [300, 2]
        logits_np = logits[0].detach().cpu().numpy()

        # 使用sigmoid将logits转换为概率，而不是softmax
        # sigmoid(x) = 1 / (1 + exp(-x))
        probs = 1 / (1 + np.exp(-logits_np))

        print(probs.max())
        # 提取前景（无人机）的概率，形状为 [300]
        drone_scores = probs[:, 1]
        
        # 3. 坐标转换：归一化cxcywh -> 增强图像素坐标xyxy
        boxes_cxcywh_pixels = boxes_cxcywh_norm * np.array([img_w, img_h, img_w, img_h])
        cx, cy, w, h = boxes_cxcywh_pixels.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(int)

        # 4. 分组可视化
        img_fg = img_np.copy()
        img_bg = img_np.copy()
        for i,b in enumerate(boxes_xyxy):
            lbl = labels_np[i]
            x1, y1, x2, y2 = b
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1: continue

            color = (0, 0, 255) if lbl == fg_cluster_idx else (255, 0, 0) # 前景红色，背景蓝色
            thickness = 2
            
            score = drone_scores[i]
            score_text = f"{score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(score_text, font, font_scale, text_thickness)
            text_x = x1
            text_y = y1 - text_height - baseline if y1 > text_height + baseline else y1 + text_height + baseline
            
            
            if lbl == fg_cluster_idx:
                cv2.rectangle(img_fg, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img_fg, score_text, (text_x, text_y), font, font_scale, color, text_thickness)
            else:
                cv2.rectangle(img_bg, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img_bg, score_text, (text_x, text_y), font, font_scale, color, text_thickness)
                
        # 5. 保存结果
        layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        save_path_fg = os.path.join(layer_dir, f"vis_foreground_{suffix}.jpg")
        save_path_bg = os.path.join(layer_dir, f"vis_background_{suffix}.jpg")
        
        cv2.imwrite(save_path_fg, img_fg)
        cv2.imwrite(save_path_bg, img_bg)
        print(f"Saved visualization for layer {layer_idx} with suffix '{suffix}' to {layer_dir}")
        
         # --- 可视化并保存GT框 ---
        # 1. 获取GT框数据
        gt_boxes_cxcywh_norm = targets[0]['boxes'].detach().cpu().numpy()
        
        # 2. 坐标转换：归一化cxcywh -> 增强图像素坐标xyxy
        # 这里的img_h和img_w在_draw_and_save函数外部已经定义，可以直接使用
        gt_boxes_cxcywh_pixels = gt_boxes_cxcywh_norm * np.array([img_w, img_h, img_w, img_h])
        cx, cy, w, h = gt_boxes_cxcywh_pixels.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        gt_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1).astype(int)

        # 3. 在原始图像副本上绘制GT框
        img_gt = img_np.copy() # 使用之前已经准备好的img_np
        gt_color = (0, 255, 0) # 使用绿色表示GT，以示区别
        gt_thickness = 2
        
        for b in gt_boxes_xyxy:
            x1, y1, x2, y2 = b
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue # 跳过无效框

            cv2.rectangle(img_gt, (x1, y1), (x2, y2), gt_color, gt_thickness)

        # 4. 保存GT可视化结果
        save_path_gt = os.path.join(layer_dir, "vis_ground_truth.jpg")
        cv2.imwrite(save_path_gt, img_gt)
        print(f"Saved ground truth visualization for layer {layer_idx} to {layer_dir}")


    # --- 主函数逻辑：调用辅助函数两次 ---
    
    # 第一次调用：可视化一阶段的分组情况
    _draw_and_save(labels, suffix="stage1")
    
    # 第二次调用：可视化二阶段的分组情况
    _draw_and_save(new_labels, suffix="stage2")
    



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
            
            
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 CSAF_enable=True,
                 ):
        super(TransformerDecoderLayer, self).__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        

        # cross attention
        if CSAF_enable:
            self.cross_attn = MSDeformableAttention(d_model, n_head, 1, n_points)    #n_levels
        else:
            self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)  
        # print(f'cross_attn params: {sum(p.numel() for p in self.cross_attn.parameters()) / 1e6:.4f} M')
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,            #[b,500,c]
                reference_points,   #[b,500,1,4]
                memory,     #[b,sum_hw,c]
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,   #无
                query_pos_embed=None):      #[b,500,c]作为query的位置嵌入
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points,               #[b,500,1,4] 
            memory,                         #[b,hw,256]
            memory_spatial_shapes,          #[[80, 80], [40, 40], [20, 20]]
            memory_mask)                    # None
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# class IGF(nn.Module):
#     def __init__(self, dim=256, dropout=0.):
#         super().__init__()
#         self.IGF = GatedResidualBlock1d(dim=dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)
#     def forward(self, x, key_padding_mask=None):
#         # x: [B, C, L] or [B, L, C] 视你的实现而定
#         out = self.IGF(x, key_padding_mask)
#         out = self.dropout(out)
#         out = self.norm(out)
#         return out
    
class TransformerDecoder(nn.Module):    #参考点和偏移量全是预测得到的
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, GQI_enable=True, GQI_num_layers=3,
                 ifexport=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        
        # GQI
        self.GQI_num_layers = GQI_num_layers
        self.GQI_enable = GQI_enable
        if GQI_enable:
            self.conv_mixer = nn.ModuleList([           # IGF
                GatedResidualBlock1d(dim=256)
                for _ in range(self.GQI_num_layers)
            ])
            # self.dropout = nn.Dropout(0.)
            # self.norm = nn.LayerNorm(256)
            
            # self.conv_mixer = nn.ModuleList([
            #     IGF(dim=256, dropout=0.)
            #     for _ in range(self.GQI_num_layers)
            # ])
        
        #export
        self.ifexport = ifexport
        
        
        
    def forward(self,
                tgt,                #合并后的[b,500,256]        在解码器中一直是 嵌入 的身份
                ref_points_unact,   #合并后的[b,500,4]          在解码器中一直是 坐标预测（参考点） 的身份
                memory,             #[b,sum_hw,c]
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                samples = None,
                targets = None):  #无
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        
        # B = output.size(0)
        # M_snapshot = self._memory_update(encoder_tokens=memory, B=B)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)       #[b,500,1,4] 这个是输入参考点（DDETR中是位置query直接预测的），不同于迭代更新的参考点（相当于query的位置编码，或者叫位置query）。在DDETR中前者是后者预测的
            query_pos_embed = query_pos_head(ref_points_detach)     #[b,500,4] -> [b,500,256]表示query的位置嵌入,
            
            # output[:, -300:, :] = self._masa(output[:, -300:, :], M_snapshot, B) 
            if i >= 3 - self.GQI_num_layers and self.GQI_enable:
                group_logits = score_head[i](output)
                if i==0:
                    inter_ref_bbox =  F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach)) #预测框
                output  = grouped_self_conv(output,group_logits, self.conv_mixer[i - 3 + self.GQI_num_layers],
                                             inter_ref_bbox, 300, ifexport=self.ifexport)
                # output = output + self.dropout(output2)
                # output = self.norm(output)

                # visualize_groups_on_image(inter_ref_bbox[:,-300:,:],labels, samples,"./vis_CGA_results",i , new_labels_b, fg_idx, targets,group_logits[:,-300:,:])
                
            output = layer(output,  ref_points_input, memory,        #解码器层  [b,500,256]
                           memory_spatial_shapes, memory_level_start_index, 
                           attn_mask, memory_mask, query_pos_embed,)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))   #bbox预测结果漂移量+参考点 [b,500,4]

            #查询最后预测 类别和BBOX偏移量
            if self.training:
                dec_out_logits.append(score_head[i](output))    #类别预测
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)       #bbox预测
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            
            #参考点更新。上一层的预测结果作为下一层的参考点
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox    #推理就不detach

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
    #[6,b,500,4]  [6,b,500,2]
    


@register()
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],       #[256,256,256]
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 GQI_enable=True,
                 GQI_num_layers=3,
                 CSAF_enable=True,
                 max_feat_c = 24,
                 ifexport=False,
                 eps=1e-2, 
                 aux_loss=True,
                 version='v1'):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes  #2
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        # CSAF可视化
        self.save_token_heatmap = False
        self.token_heatmap_dir = "./feature_token_vis"
        self.token_heatmap_step = "vis"

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, CSAF_enable)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx, GQI_enable, GQI_num_layers,ifexport)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        if version == 'v1':
            self.enc_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim,)
            )
        else:
            self.enc_output = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(hidden_dim, hidden_dim)),
                ('norm', nn.LayerNorm(hidden_dim,)),
            ]))

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        # 最大特征图
        self.CSAF_enable = CSAF_enable
        if CSAF_enable:
            self.max_feat_proj = nn.Sequential(
                    nn.Conv2d(max_feat_c, hidden_dim, kernel_size=1, bias=False),       #resnset18是64，mbv3L是24, faster是40
                    nn.BatchNorm2d(hidden_dim))
            if ifexport:
                self.Fuse = CrossScaleAlignFuseONNX(256)
            else:
                self.Fuse = CrossScaleAlignFuse(256)
        
        # self.HFP = HFP(256)       #代替CSAF
        
        
        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.03)        #0.01
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()       #输入映射：1x1卷积+归一化：调整C
        
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),   #256 -> 256
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):   #3-3=0
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features   3个特征图做映射
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]     #[b,c,h,w]
        
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs    展平特征图列表，尺寸列表，起始下标列表
        feat_flatten = []           #[b,hw,c]
        spatial_shapes = []         #[[h,w], [], []]
        level_start_index = [0, ]       #[0, hw0, hw0+hw1, hw0+hw1+hw2]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)        #[b,sum_hw,c]
        level_start_index.pop()                             #[0, hw0, hw0+hw1]
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,       #0.05偏大了点可能
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:  #推理的时候？
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []    
        for lvl, (h, w) in enumerate(spatial_shapes):   #每个像素中心一个anchor
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')    #grid_y，grid_x都是[h,w],沿着y/x方向递增
            grid_xy = torch.stack([grid_x, grid_y], -1)     #合成坐标[h,w,2]
            valid_WH = torch.tensor([w, h]).to(dtype)   #[h,w]
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH   #让anchor落在网格中心并归一化   [1,h,w,2]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)    #BBOX的宽高 [b,h,w,2] 。越高层越大:0.05->0.1->0.2.低层小锚框负责小目标，高层负责大锚框
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))   #[b,h,w,4] -> [b,hw,4]

        anchors = torch.concat(anchors, 1).to(device)   #[b,sum_hw,4]
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)    #[b,sum_hw,1] 检查xyhw是否都在区间(eps,1-eps)
        anchors = torch.log(anchors / (1 - anchors))    #坐标变换到sigmoid inverse
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)   #合法不变，不合法令为inf

        return anchors, valid_mask      #[b,sum_hw,4]为[cx,cy,w,h]形式的锚点   [b,sum_hw,1] 


    def _get_decoder_input(self,
                           memory,                      #[b, sum_hw, c] 混合编码器学到的
                           spatial_shapes,              #[[h, w],[],[]]
                           denoising_class=None,        #[b, num_denoising, 256]
                           denoising_bbox_unact=None):  #[b, num_denoising, 4]
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)  ##[b,sum_hw,4]   [b,sum_hw,1] 
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory) #线性层 [b,sum_hw,c]
        #编码器的输出预测
        enc_outputs_class = self.enc_score_head(output_memory)  # 线性层预测类别
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors   # MLP预测坐标偏移    anchors相当于参考点

        #根据类别预测值topk筛选
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)     #[bs,sum_hw,2]->[bs,sum_hw]->[bs,300] 
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))   #[b,300,4]取300个作为参考点

        enc_topk_bboxes = F.sigmoid(reference_points_unact) #参考点的激活版，取300个坐标预测
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)  #[b,500,4]  合并DN坐标查询
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))     #[bs,300,2]取300个类别预测

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])) #混合编码器输出，[bs,300,c] 内容查询
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1) #合并DN类别查询

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits    #1/2取500个,3/4取300个（经过预测的）
#[b, 500, 256], [b, 500, 4], [b, 300, 4], [b, 300, 2]   前两个用于解码器，后两个用于aux_LOSS。后3个都是预测，第一个是嵌入



    def forward(self, feats, targets=None, feat_max = None, samples=None):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)    #展平特征图和2列表
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                label_noise_ratio=self.label_noise_ratio,   #0.5
                    box_noise_scale=self.box_noise_scale, ) #1.0
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)  #解码器输入    
            
        # CSAF前特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                [feat_max, feats[-1]],
                samples, 
                stage_name="before_CSAF",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["H2", "H5"],
                step=getattr(self, "token_heatmap_step", None),
            )

        # CSAF 极尺度融合
        if self.CSAF_enable:
            feat_max = self.max_feat_proj(feat_max)
            feat_fuse = self.Fuse(feat_max,feats[-1])
            # feat_fuse = self.HFP(feat_max)
            b,c,h,w = feat_fuse.shape
            memory =  feat_fuse.permute(0, 2, 3, 1).reshape(b, h * w, c)            #[b,hw,c]
            spatial_shapes = [[h,w]]
            level_start_index = 0
        
        
        # CSAF后特征可视化
        if getattr(self, "save_token_heatmap", False):
            _save_feat_token_heatmaps(
                [feat_fuse],
                samples, 
                stage_name="after_CSAF",
                save_dir=getattr(self, "token_heatmap_dir", "./feature_token_vis"),
                names=["J2"],
                step=getattr(self, "token_heatmap_step", None),
            )
            
        # decoder
        out_bboxes, out_logits = self.decoder(                  #[6,b,500,4],[6,b,500,2]
            target,                 #合并后的类别查询
            init_ref_points_unact,  #合并后的bbox查询
            memory,             #memory
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,     #bbox预测 MLP
            self.dec_score_head,    #类别预测 线性层
            self.query_pos_head,    #MLP
            attn_mask=attn_mask,
            samples= samples,
            targets = targets,
            )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)     #分成[6,b,300,4]和[6,b,200,4]
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]} #取最后解码器的输出

        if self.training and self.aux_loss: #前5层解码器输出 + DN输出
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])       #  out['aux_outputs'][0]  - [4]
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))     #out['aux_outputs'][5]  编码器的输出
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)    ##  out['dn_aux_outputs'][0]  - [5]
                out['dn_meta'] = dn_meta

        return out 


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]      #把两个tensor拼成2元素字典



if __name__ == "__main__":
    self_attn = nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
    cross_attn = MSDeformableAttention(256, 8, 3, 4)  
    
    x = torch.randn([1,300,256])
    reference_points = torch.randn([1,300,1,4])
    memory = torch.randn([1,8400,256])
    memory_spatial_shapes = torch.randn([[80, 80], [40, 40], [20, 20]])
    memory_mask = None
    
    output1 = self_attn(x, x, x)
    output2 = cross_attn(x,
                         reference_points,          #[b,500,1,4] 
                         memory,                    #[b,hw,256]
                         memory_spatial_shapes,     #[[80, 80], [40, 40], [20, 20]]
                         memory_mask
                         )
    
    
