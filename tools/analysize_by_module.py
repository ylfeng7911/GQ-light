import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig



def print_decoder_submodule_params(model, max_depth=2, show_zero=False):
    """
    统计 decoder 内部各子模块的参数量。

    Args:
        model: RTDETR / GQ-DETR model
        max_depth:
            1 表示统计 decoder 的一级子模块，例如:
                decoder.decoder
                decoder.Fuse
                decoder.input_proj ...
            2 表示统计 decoder 的二级子模块，例如:
                decoder.decoder.layers
                decoder.decoder.layers.0
                decoder.Fuse.xxx
        show_zero: 是否打印参数量为 0 的模块
    """

    print("\n===== Decoder Submodule Parameter Statistics =====")

    if not hasattr(model, "decoder"):
        print("model has no attribute: decoder")
        return

    decoder = model.decoder

    # decoder 总参数量
    decoder_total = sum(p.numel() for p in decoder.parameters())

    # 统计 decoder 下不同层级的参数
    group_params = defaultdict(int)

    for name, p in decoder.named_parameters():
        n = p.numel()

        # name 是相对于 decoder 的名字，例如:
        #   decoder.layers.0.self_attn.in_proj_weight
        #   Fuse.xxx.weight
        parts = name.split(".")

        if len(parts) == 0:
            key = "decoder"
        else:
            # 取 max_depth 层作为分组 key
            key = ".".join(parts[:max_depth])

        group_params[key] += n

    # 排序打印
    sorted_items = sorted(group_params.items(), key=lambda kv: kv[1], reverse=True)

    print(f"{'Submodule':50s} {'Params':>15s} {'Ratio in decoder':>18s}")
    print("-" * 90)

    for key, n in sorted_items:
        if n == 0 and not show_zero:
            continue
        ratio = n / decoder_total * 100 if decoder_total > 0 else 0
        print(f"decoder.{key:42s} {n/1e6:12.3f} M {ratio:15.2f}%")

    print("-" * 90)
    print(f"{'decoder total':50s} {decoder_total/1e6:12.3f} M {100.00:15.2f}%")
    
    
    
def print_module_params(model):
    total_params = 0
    # 四个分组
    params_backbone = 0
    params_enc_A = 0   # input_proj + encoder + decoder.Fuse
    params_enc_B = 0   # lateral + fpn + downsample + pan
    params_decoder = 0 # decoder (excluding Fuse)

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n

        # ---- 1. Backbone
        if name.startswith("backbone."):
            params_backbone += n

        # ---- 2. Encoder Part A
        elif (
            name.startswith("encoder.input_proj.")
            or name.startswith("encoder.encoder.")
            or name.startswith("decoder.Fuse.")
        ):
            params_enc_A += n

        # ---- 3. Encoder Part B (FPN + PAN)
        elif (
            name.startswith("encoder.lateral_convs.")
            or name.startswith("encoder.fpn_blocks.")
            or name.startswith("encoder.downsample_convs.")
            or name.startswith("encoder.pan_blocks.")
        ):
            params_enc_B += n

        # ---- 4. Decoder (excluding Fuse)
        elif name.startswith("decoder."):
            params_decoder += n

        else:
            print("extra!!!!")

    print("\n===== Parameter Statistics (4-part split) =====")
    print(f"Backbone              : {params_backbone/1e6:.3f} M")
    print(f"Encoder Part-A (CSAF) : {params_enc_A/1e6:.3f} M")
    print(f"Encoder Part-B (FPN)  : {params_enc_B/1e6:.3f} M")
    print(f"Decoder (w/o Fuse)    : {params_decoder/1e6:.3f} M")
    print("-" * 45)
    print(f"Sum of 4 parts        : {(params_backbone + params_enc_A + params_enc_B + params_decoder)/1e6:.3f} M")
    print(f"Total model params    : {total_params/1e6:.3f} M")
        


def safe_matmul_flop(inputs, outputs):
    return 0

def _get_inclusive(flops_by_module: dict, key: str) -> float:
    """
    取 flops_by_module[key]，如果不存在就返回 0
    """
    return float(flops_by_module.get(key, 0.0))

def print_4part_gflops_inclusive(model, input_shape=(1,3,640,640), device="cuda:0", top_other=15):
    # model.eval().to(device)
    model.deploy().to(device)
    x = torch.randn(*input_shape, device=device)

    analyzer = FlopCountAnalysis(model, x)
    analyzer.set_op_handle("aten::matmul", safe_matmul_flop)
    analyzer.set_op_handle("aten::bmm", safe_matmul_flop)

    flops_by_module = analyzer.by_module()  # inclusive
    total = float(analyzer.total())

    # ---- 你的 4 块（用“子树根”的 inclusive FLOPs）
    backbone = _get_inclusive(flops_by_module, "backbone")

    # Encoder-A: input_proj + encoder + decoder.Fuse
    encA = (
        _get_inclusive(flops_by_module, "encoder.input_proj")
        + _get_inclusive(flops_by_module, "encoder.encoder")
        + _get_inclusive(flops_by_module, "decoder.Fuse")
    )

    # Encoder-B: lateral + fpn + downsample + pan
    encB = (
        _get_inclusive(flops_by_module, "encoder.lateral_convs")
        + _get_inclusive(flops_by_module, "encoder.fpn_blocks")
        + _get_inclusive(flops_by_module, "encoder.downsample_convs")
        + _get_inclusive(flops_by_module, "encoder.pan_blocks")
    )

    # Decoder(w/o Fuse) = decoder - decoder.Fuse
    decoder_total = _get_inclusive(flops_by_module, "decoder")
    fuse = _get_inclusive(flops_by_module, "decoder.Fuse")
    decoder_wo_fuse = max(decoder_total - fuse, 0.0)

    parts_sum = backbone + encA + encB + decoder_wo_fuse
    other = max(total - parts_sum, 0.0)

    print("\n===== 4-part GFLOPs (inclusive roots; sum matches total) =====")
    print(f"Input shape: {input_shape}\n")
    print(f"{'Backbone':30s}: {backbone/1e9:8.3f} G")
    print(f"{'Encoder-A (input_proj+encoder+Fuse)':30s}: {encA/1e9:8.3f} G")
    print(f"{'Encoder-B (FPN+PAN)':30s}: {encB/1e9:8.3f} G")
    print(f"{'Decoder (w/o Fuse)':30s}: {decoder_wo_fuse/1e9:8.3f} G")
    print(f"{'Other (residual)':30s}: {other/1e9:8.3f} G")
    print("-"*55)
    print(f"{'Sum of parts':30s}: {(parts_sum + other)/1e9:8.3f} G")
    print(f"{'Total (fvcore)':30s}: {total/1e9:8.3f} G")

    # ---- 帮你定位 “Other” 到底是谁在吃 FLOPs
    # 把不属于你四块根模块子树的“根级别大户”打印出来
    tracked_roots = {
        "backbone",
        "encoder.input_proj", "encoder.encoder",
        "encoder.lateral_convs", "encoder.fpn_blocks", "encoder.downsample_convs", "encoder.pan_blocks",
        "decoder", "decoder.Fuse"
    }

    # 取 flops_by_module 中 top 的条目（按 FLOPs 排序）
    top = sorted(flops_by_module.items(), key=lambda kv: float(kv[1]), reverse=True)

    print(f"\nTop-{top_other} modules by inclusive FLOPs (for debugging 'Other'):")
    shown = 0
    for name, f in top:
        # 跳过我们明确统计的根模块
        if name in tracked_roots:
            continue
        if float(f) <= 0:
            continue
        print(f"  {name:40s}  {float(f)/1e9:8.3f} G")
        shown += 1
        if shown >= top_other:
            break
        
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="/home/fyl/workspace_fyl/GQ-light/configs/gqdetr/gqdetr_fasternet_6x.yml"      #gqdetr_r18vd_6x_coco.yml
    )
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, device=args.device)
    model = cfg.model


    # 计算分组参数量
    print_module_params(model)
    
    # 更详细统计 decoder 内部参数量
    print_decoder_submodule_params(model, max_depth=2)
    
    # 计算分组计算量
    print_4part_gflops_inclusive(model, input_shape=(1,3,640,640), device="cuda:0")
    #13.7M, 38G
    #17.6M, 42G
    #22.8M, 52G