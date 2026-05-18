import torch
import torch.nn.functional as F
import torch.nn as nn


class GNNMixer(nn.Module):
    """
    轻量图神经网络 (GAT风格) 用于 TGT 分组交互
    输入: [B, L, C]
    输出: [B, L, C]
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # x: [B, L, C]
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, C//H]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, H, L]

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].expand(-1, self.heads, L, -1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)  # [B, L, C]

        out = self.proj(out)
        out = self.dropout(out)
        return self.norm(x + out)   # 残差+归一化



# 定义一个简单的 1D 卷积交互模块
class Conv1dGroupMixer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        """
        dim: 特征维度 (C)
        kernel_size: 卷积核大小（建议 3 或 5）
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,   # 如果想分通道做卷积，可以设为 dim
            bias=False
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, L, C]
        key_padding_mask: [B, L]，True 表示该位置是 padding
        """
        B, L, C = x.shape
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, C]
        out = self.norm(out)

        if key_padding_mask is not None:
            # 把 padding 的位置置零，避免污染
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return out
    


class GatedConv1d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim*2, kernel_size, padding=kernel_size//2)

    def forward(self, x):  # x: [B, C, L]
        out = self.conv(x)               # [B, 2C, L]
        out, gate = out.chunk(2, dim=1)  # 划分为信号和门
        return out * torch.sigmoid(gate) # 门控调节

#IGF
class MultiScaleGatedConv1d(nn.Module):
    def __init__(self, dim, kernels=(3,7), dilations=(1,4)):        #kernels=(3, 7), dilations=(1, 4)
        super().__init__()
        #原版
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(dim, dim * 2, k, padding=k // 2) for k in kernels
        # ])
        # self.proj = nn.Conv1d(dim * 2 * len(kernels), dim * 2, 1)  # 融合多尺度
        
        #空洞卷积版本
        self.convs = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            # <--- 2. 动态计算padding以保持序列长度不变
            padding = (k - 1) * d // 2              # 1,4,12
            
            # <--- 3. 将dilation和padding传递给Conv1d
            self.convs.append(
                nn.Conv1d(dim, dim * 2, k, stride=1, dilation=d, padding=padding)
            )
            
        self.proj = nn.Conv1d(dim * 2 * len(kernels), dim * 2, 1)

        
    def forward(self, x):  # x: [B, C, L]
        outs = [conv(x) for conv in self.convs]   # 多尺度卷积 [B, 2C, L]
        out = torch.cat(outs, dim=1)              # [B, 2C*len(kernels), L]
        out = self.proj(out)                      # 压缩回 [B, 2C, L]
        signal, gate = out.chunk(2, dim=1)        # 划分为信号和门
        return signal * torch.sigmoid(gate)       # 门控调节


class GatedResidualBlock1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gated_conv = MultiScaleGatedConv1d(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):  # [B, L, C]
        residual = x
        x = x.transpose(1, 2)        # [B, C, L]
        out = self.gated_conv(x)     # [B, C, L]
        out = out.transpose(1, 2)    # [B, L, C]
        out = self.norm(out)
        
        if key_padding_mask is not None:
            # 扩展成 [B, L, 1]，使其能广播到 [B, L, C]
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return residual + out        # 残差加法



def boxes_iou(boxes1, boxes2):
    """
    计算 IoU（适用于 cx, cy, w, h 格式，归一化坐标）
    boxes1: [N, 4], boxes2: [M, 4]
    返回: [N, M] 的 IoU 矩阵
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    # cxcywh -> x1y1x2y2
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    # clamp防止出现负值（归一化下安全）
    boxes1_x1 = boxes1_x1.clamp(0, 1)
    boxes1_y1 = boxes1_y1.clamp(0, 1)
    boxes1_x2 = boxes1_x2.clamp(0, 1)
    boxes1_y2 = boxes1_y2.clamp(0, 1)

    boxes2_x1 = boxes2_x1.clamp(0, 1)
    boxes2_y1 = boxes2_y1.clamp(0, 1)
    boxes2_x2 = boxes2_x2.clamp(0, 1)
    boxes2_y2 = boxes2_y2.clamp(0, 1)

    # 计算交集区域
    lt_x = torch.max(boxes1_x1[:, None], boxes2_x1[None, :])  # [N, M]
    lt_y = torch.max(boxes1_y1[:, None], boxes2_y1[None, :])
    rb_x = torch.min(boxes1_x2[:, None], boxes2_x2[None, :])
    rb_y = torch.min(boxes1_y2[:, None], boxes2_y2[None, :])

    inter_w = (rb_x - lt_x).clamp(min=0)
    inter_h = (rb_y - lt_y).clamp(min=0)
    inter = inter_w * inter_h  # [N, M]

    area1 = (boxes1_x2 - boxes1_x1).clamp(min=0) * (boxes1_y2 - boxes1_y1).clamp(min=0)
    area2 = (boxes2_x2 - boxes2_x1).clamp(min=0) * (boxes2_y2 - boxes2_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    iou = torch.zeros_like(inter)
    valid = union > 0
    iou[valid] = inter[valid] / union[valid]
    return iou


    
def kmeans_torch(x, K=2, num_iters=10, eps=1e-6, seed=0):
    """
    简单的 KMeans (单样本). x: [N, C]
    返回: labels [N] (long), centers [K, C]
    说明: 适用于 N ~ 300, K 小（2~8）。
    """
    device = x.device
    N, C = x.shape
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((K, C), device=device)

    K = min(K, max(1, N))
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    # 随机初始化 centers 从 x 采样
    if N >= K:
        perm = torch.randperm(N, generator=rng, device=device)[:K]
        centers = x[perm].clone()
    else:
        centers = torch.zeros((K, C), device=device)
        centers[:N] = x.clone()

    labels = torch.zeros(N, dtype=torch.long, device=device)
    for it in range(num_iters):
        # 计算平方距离: ||x||^2 - 2 x c^T + ||c||^2
        x_norm2 = (x * x).sum(dim=1, keepdim=True)           # [N,1]
        c_norm2 = (centers * centers).sum(dim=1, keepdim=True).T  # [1,K]
        dist = x_norm2 - 2.0 * (x @ centers.T) + c_norm2     # [N,K]
        new_labels = torch.argmin(dist, dim=1)               # [N]          计算欧式距离，和哪个中心近归哪个
        if it > 0 and torch.equal(new_labels, labels):
            break
        labels = new_labels                                                 #重新分配标签
        centers_prev = centers.clone()
        # 更新 centers
        for k in range(K):
            mask = (labels == k)                                            #更新2个簇的中心，再次迭代
            if mask.any():
                centers[k] = x[mask].mean(dim=0)
            else:
                # 空簇：重新随机初始化为一个样本点，避免中心为 zero 导致退化
                centers[k] = x[torch.randint(0, N, (1,), device=device)].squeeze(0)
        if torch.max((centers - centers_prev).abs()) < eps:
            break
    return labels, centers


def deterministic_2means(x, eps=1e-6):
    """
    Deterministic substitute for KMeans (K=2)
    x: [N, C]
    return:
        labels: [N] (0 or 1)
        centers: [2, C]
    """
    device = x.device
    N, C = x.shape

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty((2, C), device=device),
        )

    # 1. mean center
    mu = x.mean(dim=0, keepdim=True)          # [1, C]
    xc = x - mu                               # [N, C]

    # 2. dominant direction (no eigen, ONNX-safe)
    # use sum of squared values as a cheap proxy
    # direction ~ sum_i |x_i - mu|
    direction = xc.abs().mean(dim=0)          # [C]
    direction = direction / (direction.norm() + eps)

    # 3. projection
    proj = xc @ direction                     # [N]

    # 4. deterministic split
    thresh = proj.mean()                      # scalar
    labels = (proj > thresh).long()           # [N]

    # 5. compute centers
    mask0 = labels == 0
    mask1 = ~mask0

    c0 = torch.where(
        mask0.unsqueeze(1),
        x,
        torch.zeros_like(x),
    ).sum(dim=0) / (mask0.sum().clamp(min=1).float())

    c1 = torch.where(
        mask1.unsqueeze(1),
        x,
        torch.zeros_like(x),
    ).sum(dim=0) / (mask1.sum().clamp(min=1).float())

    centers = torch.stack([c0, c1], dim=0)    # [2, C]

    return labels, centers


# 单阶段版本
def grouped_self_con_one_stage(
    tgt,                       # [B, Q, C]
    enc_topk_logits,           # [B, Q, num_classes]
    conv_mixer,
    ref_boxes,
    num_queries=300,
    K=2,                       # 聚类簇数
    kmeans_iters=10,
    min_keep=1,
    min_keep_bg=1,
):
    """
    基于聚类的分组卷积交互：
    - 在 tgt 的最后 num_queries 个 queries 上做聚类
    - 用 enc_topk_logits 的平均前景概率判定前/背景簇
    - 对每个簇做 1D 卷积交互
    """
    B, Q, C = tgt.shape
    device = tgt.device
    M = num_queries

    tgt0 = tgt[:, -M:, :].contiguous()
    logits0 = enc_topk_logits[:, -M:, :].contiguous()

    if logits0.shape[-1] == 1:
        scores_all = torch.sigmoid(logits0.view(B, M))
    else:
        fg_idx = 1
        scores_all = torch.sigmoid(logits0[..., fg_idx])

    tgt0_new = torch.zeros_like(tgt0)

    tgt_fg_list, tgt_bg_list = [], []
    idx_fg_list, idx_bg_list = [], []
    len_fg_list, len_bg_list = [], []

    #每个样本：做聚类，得到前后景tgt列表、前后景idx列表
    for b in range(B):
        x = tgt0[b]
        feats = F.normalize(x.detach(), p=2, dim=1)

        K_b = min(K, max(1, M))
        labels, centers = kmeans_torch(feats, K=K_b, num_iters=kmeans_iters, seed=b+1)  

        cluster_scores = []
        for k in range(K_b):
            mask_k = (labels == k)
            if mask_k.sum() == 0:
                cluster_scores.append(torch.tensor(-1.0, device=device))
            else:
                cluster_scores.append(scores_all[b][mask_k].mean().detach())
        cluster_scores = torch.stack(cluster_scores)

        fg_cluster_idx = int(torch.argmax(cluster_scores).item())
        fg_mask_b = (labels == fg_cluster_idx)
        bg_mask_b = ~fg_mask_b

        if fg_mask_b.sum() < min_keep:
            _, topk_idx = torch.topk(scores_all[b], min(min_keep, M), dim=0)
            new_fg = torch.zeros_like(fg_mask_b)
            new_fg[topk_idx] = True
            fg_mask_b = new_fg
            bg_mask_b = ~fg_mask_b

        if bg_mask_b.sum() < min_keep_bg:
            lowest = torch.argmin(scores_all[b])
            bg_mask_b[lowest] = True
            fg_mask_b[lowest] = False

        idx_fg = torch.nonzero(fg_mask_b, as_tuple=False).squeeze(-1)
        idx_bg = torch.nonzero(bg_mask_b, as_tuple=False).squeeze(-1)
        tgt_fg = x[idx_fg] if idx_fg.numel() > 0 else x.new_zeros((0, C))
        tgt_bg = x[idx_bg] if idx_bg.numel() > 0 else x.new_zeros((0, C))

        tgt_fg_list.append(tgt_fg)
        tgt_bg_list.append(tgt_bg)
        idx_fg_list.append(idx_fg)
        idx_bg_list.append(idx_bg)
        len_fg_list.append(tgt_fg.size(0))
        len_bg_list.append(tgt_bg.size(0))

    #分别对前后景：每个样本对应的tgt补零到batch中的最长长度
    max_fg = max(len_fg_list) if len_fg_list else 0
    max_bg = max(len_bg_list) if len_bg_list else 0

    def pad_group_list(group_list, max_len):
        padded = []
        for x in group_list:
            cur = x
            pad = max_len - cur.size(0)
            if pad == 0:
                padded.append(cur)
            else:
                if cur.size(0) == 0:
                    padded.append(cur.new_zeros((max_len, C)))
                else:
                    padded.append(F.pad(cur, (0, 0, 0, pad)))
        return torch.stack(padded, dim=0) if len(padded) > 0 else tgt0.new_zeros((B, 0, C))

    padded_tgt_fg = pad_group_list(tgt_fg_list, max_fg) if max_fg > 0 else tgt0.new_zeros((B, 0, C))
    padded_tgt_bg = pad_group_list(tgt_bg_list, max_bg) if max_bg > 0 else tgt0.new_zeros((B, 0, C))

    if max_fg > 0:
        lengths_fg = torch.tensor(len_fg_list, device=device)
        key_pad_fg = (torch.arange(max_fg, device=device).unsqueeze(0).expand(B, -1) >= lengths_fg.unsqueeze(1)).to(torch.bool)#掩码，T表示pad部分
        padded_tgt_fg_after = conv_mixer(padded_tgt_fg, key_padding_mask=key_pad_fg)
    else:
        padded_tgt_fg_after = padded_tgt_fg

    if max_bg > 0:
        lengths_bg = torch.tensor(len_bg_list, device=device)
        key_pad_bg = (torch.arange(max_bg, device=device).unsqueeze(0).expand(B, -1) >= lengths_bg.unsqueeze(1)).to(torch.bool)
        padded_tgt_bg_after = conv_mixer(padded_tgt_bg, key_padding_mask=key_pad_bg)
        padded_tgt_bg_after = padded_tgt_bg
    else:
        padded_tgt_bg_after = padded_tgt_bg

    for b in range(B):
        lf = len_fg_list[b]
        lb = len_bg_list[b]
        if lf > 0:
            tgt0_new[b, idx_fg_list[b]] = padded_tgt_fg_after[b, :lf]
        if lb > 0:
            tgt0_new[b, idx_bg_list[b]] = padded_tgt_bg_after[b, :lb]

    tgt_new = torch.cat([tgt[:, :-M, :], tgt0_new], dim=1)
    return tgt_new



# 两阶段版本
def grouped_self_conv(
    tgt,                       # [B, Q, C]
    enc_topk_logits,           # [B, Q, num_classes]
    conv_mixer,
    ref_boxes,
    num_queries=300,
    K=2,                       # 聚类簇数
    kmeans_iters=10,
    min_keep=1,
    min_keep_bg=1,
    ifexport=False,
):
    """
    基于聚类的分组卷积交互：
    - 在 tgt 的最后 num_queries 个 queries 上做聚类
    - 用 enc_topk_logits 的平均前景概率判定前/背景簇
    - 对每个簇做 1D 卷积交互
    """
    second_stage_iou_thresh = 0.6           #原先是0.3
    B, Q, C = tgt.shape
    device = tgt.device
    M = num_queries
    
    tgt0 = tgt[:, -M:, :].contiguous()
    logits0 = enc_topk_logits[:, -M:, :].contiguous()

    if logits0.shape[-1] == 1:
        scores_all = torch.sigmoid(logits0.view(B, M))
    else:
        fg_idx = 1
        scores_all = torch.sigmoid(logits0[..., fg_idx])

    tgt0_new = torch.zeros_like(tgt0)

    tgt_fg_list, tgt_bg_list = [], []
    idx_fg_list, idx_bg_list = [], []
    len_fg_list, len_bg_list = [], []

    # 1.每个样本：做聚类，得到前后景tgt列表、前后景idx列表
    for b in range(B):
        x = tgt0[b]
        score_b = scores_all[b] 
        feats = F.normalize(x.detach(), p=2, dim=1)         #归一化，固定通道c,对不同的q进行归一化（？？？）

        K_b = min(K, max(1, M))
        if ifexport:
            labels, centers = deterministic_2means(feats)
        else:
            labels, centers = kmeans_torch(feats, K=K_b, num_iters=kmeans_iters, seed=b+1)  
        

        cluster_scores = []
        for k in range(K_b):
            mask_k = (labels == k)
            if mask_k.sum() == 0:
                cluster_scores.append(torch.tensor(-1.0, device=device))
            else:
                cluster_scores.append(scores_all[b][mask_k].mean().detach())
        cluster_scores = torch.stack(cluster_scores)

        # 判别1：score
        fg_cluster_idx = int(torch.argmax(cluster_scores).item())
        fg_mask_b = (labels == fg_cluster_idx)
        bg_mask_b = ~fg_mask_b

        if fg_mask_b.sum() < min_keep:
            _, topk_idx = torch.topk(scores_all[b], min(min_keep, M), dim=0)
            new_fg = torch.zeros_like(fg_mask_b)
            new_fg[topk_idx] = True
            fg_mask_b = new_fg
            bg_mask_b = ~fg_mask_b

        if bg_mask_b.sum() < min_keep_bg:
            lowest = torch.argmin(scores_all[b])
            bg_mask_b[lowest] = True
            fg_mask_b[lowest] = False

        idx_fg = torch.nonzero(fg_mask_b, as_tuple=False).squeeze(-1)
        idx_bg = torch.nonzero(bg_mask_b, as_tuple=False).squeeze(-1)
        tgt_fg = x[idx_fg] if idx_fg.numel() > 0 else x.new_zeros((0, C))
        tgt_bg = x[idx_bg] if idx_bg.numel() > 0 else x.new_zeros((0, C))
        
        # 判别2：box  把 background 中与 foreground box IoU高的移到 foreground（修改tgt集合和idx集合）
        # 使用 ref_boxes 对应最后 M 个 queries
        # 假定 ref_boxes 是 [B, Q_total, 4]，且 boxes 格式为 cx,cy,w,h
        if idx_bg.numel() > 0 and idx_fg.numel() > 0:
            boxes_b = ref_boxes[b, -M:, :]  # [M,4]
            # select bg boxes and fg boxes
            bg_boxes = boxes_b[idx_bg]  # [nb,4]
            fg_boxes = boxes_b[idx_fg]  # [nf,4]
            # compute IoU matrix [nb, nf]
            iou_mat = boxes_iou(bg_boxes, fg_boxes)
            # for each bg box, check max IoU with any fg box
            max_iou_per_bg, _ = iou_mat.max(dim=1)  # [nb]
            # determine which bg indices to move
            move_mask = max_iou_per_bg > float(second_stage_iou_thresh)
            if move_mask.any():
                # 接受再分类的（背景集合）下标
                move_idx_in_bg = torch.nonzero(move_mask, as_tuple=False).squeeze(-1)
                if move_idx_in_bg.dim() == 0:
                    move_idx_in_bg = move_idx_in_bg.unsqueeze(0)
                # 接受再分类的（全集）下标
                to_move_query_indices = idx_bg[move_idx_in_bg]  # global within last M queries (0..M-1)
                # new_idx_bg是原本idx_bg的子集(若全移动需要额外处理)
                if idx_bg.numel() == move_idx_in_bg.numel():    #全移动
                    # all bg moved -> bg becomes empty
                    new_idx_bg = idx_bg.new_zeros((0,), dtype=torch.long)   # 空张量[]
                else:
                    mask_keep_bg = torch.ones_like(idx_bg, dtype=torch.bool)
                    mask_keep_bg[move_idx_in_bg] = False
                    new_idx_bg = idx_bg[mask_keep_bg]

                new_idx_fg = torch.cat([idx_fg, to_move_query_indices], dim=0)

                # sort indices for stability (可选)
                new_idx_fg, _ = torch.sort(new_idx_fg)
                if new_idx_bg.numel() > 0:
                    new_idx_bg, _ = torch.sort(new_idx_bg)

                # 保证new_idx_fg一定是
                tgt_fg = x[new_idx_fg] if new_idx_fg.numel() > 0 else x.new_zeros((0, C))
                tgt_bg = x[new_idx_bg] if new_idx_bg.numel() > 0 else x.new_zeros((0, C))

                idx_fg = new_idx_fg
                idx_bg = new_idx_bg
        # ----- end second-stage -----
        
        #调试！！！
        # new_labels_b = torch.zeros(M, dtype=torch.long, device=device)
        # # 将前景query对应的label设置为1
        # if idx_fg.numel() > 0:
        #     new_labels_b[idx_fg] = fg_cluster_idx
        # if idx_bg.numel() > 0:
        #     new_labels_b[idx_bg] = 1 - fg_cluster_idx
        
        tgt_fg_list.append(tgt_fg)
        tgt_bg_list.append(tgt_bg)
        idx_fg_list.append(idx_fg)
        idx_bg_list.append(idx_bg)
        len_fg_list.append(tgt_fg.size(0))
        len_bg_list.append(tgt_bg.size(0))

    # 2.分别对前后景：每个样本对应的tgt补零到batch中的最长长度
    max_fg = max(len_fg_list) if len_fg_list else 0
    max_bg = max(len_bg_list) if len_bg_list else 0

    def pad_group_list(group_list, max_len):
        padded = []
        for x in group_list:
            cur = x
            pad = max_len - cur.size(0)
            if pad == 0:
                padded.append(cur)
            else:
                if cur.size(0) == 0:
                    padded.append(cur.new_zeros((max_len, C)))      #创建(max_len, C)的zeros，继承cur的type
                else:
                    padded.append(F.pad(cur, (0, 0, 0, pad)))
        return torch.stack(padded, dim=0) if len(padded) > 0 else tgt0.new_zeros((B, 0, C))

    padded_tgt_fg = pad_group_list(tgt_fg_list, max_fg) if max_fg > 0 else tgt0.new_zeros((B, 0, C))
    padded_tgt_bg = pad_group_list(tgt_bg_list, max_bg) if max_bg > 0 else tgt0.new_zeros((B, 0, C))

    
    # 3.补零后前后景tgt交互（带屏蔽）
    if max_fg > 0:
        lengths_fg = torch.tensor(len_fg_list, device=device)
        key_pad_fg = (torch.arange(max_fg, device=device).unsqueeze(0).expand(B, -1) >= lengths_fg.unsqueeze(1)).to(torch.bool)#掩码，T表示pad部分
        padded_tgt_fg_after = conv_mixer(padded_tgt_fg, key_padding_mask=key_pad_fg)
    else:
        padded_tgt_fg_after = padded_tgt_fg

    if max_bg > 0:
        lengths_bg = torch.tensor(len_bg_list, device=device)
        key_pad_bg = (torch.arange(max_bg, device=device).unsqueeze(0).expand(B, -1) >= lengths_bg.unsqueeze(1)).to(torch.bool)
        padded_tgt_bg_after = conv_mixer(padded_tgt_bg, key_padding_mask=key_pad_bg)
        # padded_tgt_bg_after = padded_tgt_bg
    else:
        padded_tgt_bg_after = padded_tgt_bg

    # 4.从补零tgt中取出真实长度的tgt再写回
    for b in range(B):
        lf = len_fg_list[b]
        lb = len_bg_list[b]
        if lf > 0:
            tgt0_new[b, idx_fg_list[b]] = padded_tgt_fg_after[b, :lf]
        if lb > 0:
            tgt0_new[b, idx_bg_list[b]] = padded_tgt_bg_after[b, :lb]

    tgt_new = torch.cat([tgt[:, :-M, :], tgt0_new], dim=1)
    return tgt_new
    # return tgt_new, labels, new_labels_b, fg_idx 





def grouped_self_conv_logits(
    tgt,                       # [B, Q, C]
    enc_topk_logits,           # [B, Q, num_classes]
    conv_mixer,
    ref_boxes,
    num_queries=300,
    K=2,                       # 聚类簇数
    kmeans_iters=10,
    min_keep=1,
    min_keep_bg=1,
):
    """
    基于聚类的分组卷积交互：
    - 在 tgt 的最后 num_queries 个 queries 上做聚类
    - 用 enc_topk_logits 的平均前景概率判定前/背景簇
    - 对每个簇做 1D 卷积交互
    """
    second_stage_iou_thresh = 0.3
    B, Q, C = tgt.shape
    device = tgt.device
    M = num_queries
    
    tgt0 = tgt[:, -M:, :].contiguous()
    logits0 = enc_topk_logits[:, -M:, :].contiguous()

    if logits0.shape[-1] == 1:
        scores_all = torch.sigmoid(logits0.view(B, M))
    else:
        probs = torch.softmax(logits0, dim=-1)
        fg_idx = 1 if logits0.shape[-1] > 1 else 0
        scores_all = probs[..., fg_idx]

    tgt0_new = torch.zeros_like(tgt0)

    tgt_fg_list, tgt_bg_list = [], []
    idx_fg_list, idx_bg_list = [], []
    len_fg_list, len_bg_list = [], []

    # 1.每个样本：根据每个tgt对应的logits，得到前后景tgt列表、前后景idx列表
    for b in range(B):
        x = tgt0[b]
        
        # 前景判别方式改为：logits 大于阈值 0.3 作为前景
        score_b = scores_all[b]  # [M]
        fg_mask_b = (score_b > 0.3)
        bg_mask_b = ~fg_mask_b


        if fg_mask_b.sum() < min_keep:
            _, topk_idx = torch.topk(scores_all[b], min(min_keep, M), dim=0)
            new_fg = torch.zeros_like(fg_mask_b)
            new_fg[topk_idx] = True
            fg_mask_b = new_fg
            bg_mask_b = ~fg_mask_b

        if bg_mask_b.sum() < min_keep_bg:
            lowest = torch.argmin(scores_all[b])
            bg_mask_b[lowest] = True
            fg_mask_b[lowest] = False

        idx_fg = torch.nonzero(fg_mask_b, as_tuple=False).squeeze(-1)
        idx_bg = torch.nonzero(bg_mask_b, as_tuple=False).squeeze(-1)
        tgt_fg = x[idx_fg] if idx_fg.numel() > 0 else x.new_zeros((0, C))
        tgt_bg = x[idx_bg] if idx_bg.numel() > 0 else x.new_zeros((0, C))
        
        # 判别2：box  把 background 中与 foreground box IoU高的移到 foreground（修改tgt集合和idx集合）
        # 使用 ref_boxes 对应最后 M 个 queries
        # 假定 ref_boxes 是 [B, Q_total, 4]，且 boxes 格式为 cx,cy,w,h
        if idx_bg.numel() > 0 and idx_fg.numel() > 0:
            boxes_b = ref_boxes[b, -M:, :]  # [M,4]
            # select bg boxes and fg boxes
            bg_boxes = boxes_b[idx_bg]  # [nb,4]
            fg_boxes = boxes_b[idx_fg]  # [nf,4]
            # compute IoU matrix [nb, nf]
            iou_mat = boxes_iou(bg_boxes, fg_boxes)
            # for each bg box, check max IoU with any fg box
            max_iou_per_bg, _ = iou_mat.max(dim=1)  # [nb]
            # determine which bg indices to move
            move_mask = max_iou_per_bg > float(second_stage_iou_thresh)
            if move_mask.any():
                # 接受再分类的（背景集合）下标
                move_idx_in_bg = torch.nonzero(move_mask, as_tuple=False).squeeze(-1)
                if move_idx_in_bg.dim() == 0:
                    move_idx_in_bg = move_idx_in_bg.unsqueeze(0)
                # 接受再分类的（全集）下标
                to_move_query_indices = idx_bg[move_idx_in_bg]  # global within last M queries (0..M-1)
                # new_idx_bg是原本idx_bg的子集(若全移动需要额外处理)
                if idx_bg.numel() == move_idx_in_bg.numel():    #全移动
                    # all bg moved -> bg becomes empty
                    new_idx_bg = idx_bg.new_zeros((0,), dtype=torch.long)   # 空张量[]
                else:
                    mask_keep_bg = torch.ones_like(idx_bg, dtype=torch.bool)
                    mask_keep_bg[move_idx_in_bg] = False
                    new_idx_bg = idx_bg[mask_keep_bg]

                new_idx_fg = torch.cat([idx_fg, to_move_query_indices], dim=0)

                # sort indices for stability (可选)
                new_idx_fg, _ = torch.sort(new_idx_fg)
                if new_idx_bg.numel() > 0:
                    new_idx_bg, _ = torch.sort(new_idx_bg)

                # 保证new_idx_fg一定是
                tgt_fg = x[new_idx_fg] if new_idx_fg.numel() > 0 else x.new_zeros((0, C))
                tgt_bg = x[new_idx_bg] if new_idx_bg.numel() > 0 else x.new_zeros((0, C))

                idx_fg = new_idx_fg
                idx_bg = new_idx_bg
        # ----- end second-stage -----
        
        # #调试！！！
        # new_labels_b = torch.zeros(M, dtype=torch.long, device=device)
        # # 将前景query对应的label设置为1
        # if idx_fg.numel() > 0:
        #     new_labels_b[idx_fg] = fg_cluster_idx
        # if idx_bg.numel() > 0:
        #     new_labels_b[idx_bg] = 1 - fg_cluster_idx
        
        tgt_fg_list.append(tgt_fg)
        tgt_bg_list.append(tgt_bg)
        idx_fg_list.append(idx_fg)
        idx_bg_list.append(idx_bg)
        len_fg_list.append(tgt_fg.size(0))
        len_bg_list.append(tgt_bg.size(0))

    # 2.分别对前后景：每个样本对应的tgt补零到batch中的最长长度
    max_fg = max(len_fg_list) if len_fg_list else 0
    max_bg = max(len_bg_list) if len_bg_list else 0

    def pad_group_list(group_list, max_len):
        padded = []
        for x in group_list:
            cur = x
            pad = max_len - cur.size(0)
            if pad == 0:
                padded.append(cur)
            else:
                if cur.size(0) == 0:
                    padded.append(cur.new_zeros((max_len, C)))      #创建(max_len, C)的zeros，继承cur的type
                else:
                    padded.append(F.pad(cur, (0, 0, 0, pad)))
        return torch.stack(padded, dim=0) if len(padded) > 0 else tgt0.new_zeros((B, 0, C))

    padded_tgt_fg = pad_group_list(tgt_fg_list, max_fg) if max_fg > 0 else tgt0.new_zeros((B, 0, C))
    padded_tgt_bg = pad_group_list(tgt_bg_list, max_bg) if max_bg > 0 else tgt0.new_zeros((B, 0, C))

    
    # 3.补零后前后景tgt交互（带屏蔽）
    if max_fg > 0:
        lengths_fg = torch.tensor(len_fg_list, device=device)
        key_pad_fg = (torch.arange(max_fg, device=device).unsqueeze(0).expand(B, -1) >= lengths_fg.unsqueeze(1)).to(torch.bool)#掩码，T表示pad部分
        padded_tgt_fg_after = conv_mixer(padded_tgt_fg, key_padding_mask=key_pad_fg)
    else:
        padded_tgt_fg_after = padded_tgt_fg

    if max_bg > 0:
        lengths_bg = torch.tensor(len_bg_list, device=device)
        key_pad_bg = (torch.arange(max_bg, device=device).unsqueeze(0).expand(B, -1) >= lengths_bg.unsqueeze(1)).to(torch.bool)
        padded_tgt_bg_after = conv_mixer(padded_tgt_bg, key_padding_mask=key_pad_bg)
        padded_tgt_bg_after = padded_tgt_bg
    else:
        padded_tgt_bg_after = padded_tgt_bg

    # 4.从补零tgt中取出真实长度的tgt再写回
    for b in range(B):
        lf = len_fg_list[b]
        lb = len_bg_list[b]
        if lf > 0:
            tgt0_new[b, idx_fg_list[b]] = padded_tgt_fg_after[b, :lf]
        if lb > 0:
            tgt0_new[b, idx_bg_list[b]] = padded_tgt_bg_after[b, :lb]

    tgt_new = torch.cat([tgt[:, :-M, :], tgt0_new], dim=1)
    return tgt_new



if __name__ == "__main__":
    # class GroupSelfAttention(nn.Module):
    #     def __init__(self, hidden_dim=256, nhead=8, dropout=0.0):
    #         super().__init__()
    #         self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
    #         self.norm = nn.LayerNorm(hidden_dim)

    #     def forward(self, x, key_padding_mask):
    #         # x: [B, N, C]
    #         attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)   #注意key_padding_mask和attn_mask的区别
    #         x = self.norm(x + attn_out)
    #         return x

    # conv_mixer = GatedResidualBlock1d(256)
    # tgt = torch.rand([8,1,256])
    # tgt = conv_mixer(tgt)
    
    
    def print_model_parameters(model):
        total = 0
        print("== Parameter breakdown ==")
        for name, param in model.named_parameters():
            if param.requires_grad:
                count = param.numel()
                total += count
                print(f"{name:50s}: {count}")
        print(f"\nTotal trainable parameters: {total}")
        
    model = GatedResidualBlock1d(dim=256)
    print_model_parameters(model)