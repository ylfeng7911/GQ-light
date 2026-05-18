"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table
def calculate_model_complexity(model, im_data, orig_size):
    """计算模型的FLOPs、参数和激活值"""
    # 创建包装函数来处理多参数输入
    def model_wrapper(x):
        return model(x, orig_size)
    
    # 计算FLOPs
    flops = FlopCountAnalysis(model_wrapper, im_data)
    
    """计算模型的FLOPs、参数和激活值"""

    model.eval()

    # fvcore 要求传入 nn.Module，而不是 function
    inputs = (im_data, orig_size)

    # 计算FLOPs
    flops = FlopCountAnalysis(model, inputs)
    
    # 计算激活值
    acts = ActivationCountAnalysis(model_wrapper, im_data)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_all = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"Input shape: {im_data.shape}")
    print(f"Original image size: {orig_size.cpu().numpy()[0]}")
    print(f"Total FLOPs: {flops.total():,}")
    print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
    print(f"Total Activations: {acts.total():,}")
    print(f"Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Total Parameters: {total_params_all:,} ({total_params_all / 1e6:.2f} M)")
    print()
    
    # 打印详细的FLOPs分解
    print("DETAILED FLOPs BREAKDOWN:")
    print("-" * 40)
    print(flop_count_table(flops, max_depth=3))
    
    return {
        'flops': flops.total(),
        'gflops': flops.total() / 1e9,
        'activations': acts.total(),
        'params_trainable': total_params,
        'params_total': total_params_all
    }



def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)


    # 计算模型复杂度
    complexity_info = calculate_model_complexity(model, im_data, orig_size)
    
    # 执行推理（可选，验证模型正常工作）
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    
    with torch.no_grad():
        output = model(im_data, orig_size)
        print(f"Output type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"Output length: {len(output)}")
            for i, out in enumerate(output):
                print(f"Output[{i}] shape: {out.shape if hasattr(out, 'shape') else 'No shape'}")
        elif isinstance(output, dict):
            print("Output keys:", list(output.keys()))
            for k, v in output.items():
                print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        else:
            print(f"Output shape: {output.shape}")
    
    return complexity_info, output
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/gqdetr/gqdetr_r18vd_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, default='/home/fyl/workspace_fyl/GQ-light/exps_light/20251212_050003_UAVSwarm_GQDETR/best.pth')
    parser.add_argument('-f', '--im-file', type=str, default='../test_scene/00002.jpg')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')      #cpu
    args = parser.parse_args()
    main(args)
