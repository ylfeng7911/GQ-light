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


def draw(images, labels, boxes, scores, base_name = None, thrh = 0.5):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red', width=2)
            draw.text((b[0], b[1]-10), text=f"{scrs[j].item():.3f}", fill='blue', )

        print(f'result_{base_name}.jpg')
        im.save(f'result_{base_name}.jpg', quality=95)
        


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

    output= model(im_data, orig_size)
    labels, boxes, scores = output

    base_name = args.im_file.split('/')[-1].replace('.jpg', '')
    draw([im_pil], labels, boxes, scores, base_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/gqdetr/gqdetr_r18vd_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, 
                        default='/root/fengyulei/GQ-light/exps_light/20260525_010003_GQ_R18/last.pth')
    parser.add_argument('-f', '--im-file', type=str, default='../test_scene/UAVSwarm1/19-10.jpg')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')      #cpu 
    args = parser.parse_args()
    main(args)
