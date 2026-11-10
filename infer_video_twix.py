"""Export per-frame GQ-DETR detections for the C-TWiX DanceTrack loader.

Uses references/deploy/rtdetrv2_torch.py's preprocessing and deploy interface.
The deployed postprocessor returns original-resolution pixel xyxy boxes.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from src.core import YAMLConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = '/root/fengyulei/exps/20260520_102901_TT_RTC2_LOSS_ASFF_final/best.pth'


class DeployModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def load_model(config, weights, device):
    cfg = YAMLConfig(str(config), resume=str(weights))
    checkpoint = torch.load(str(weights), map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    # Same strict loading and train-to-deploy conversion as the reference.
    cfg.model.load_state_dict(state, strict=True)
    model = DeployModel(cfg).to(device).eval()
    print(f'Loaded weights (strict): {weights}')
    return model


def valid_detections(boxes, scores, width, height, threshold):
    """Clip deployed pixel xyxy boxes; keep small targets without area filtering."""
    boxes = boxes.detach().cpu().numpy().copy()
    scores = scores.detach().cpu().numpy()
    keep = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores > threshold)
    boxes, scores = boxes[keep], scores[keep]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], scores[keep]


def main(args):
    if not math.isfinite(args.score_thr) or not 0 <= args.score_thr <= 1:
        raise ValueError('--score-thr must be finite and in [0, 1]')
    video, weights, config = (Path(p).expanduser().resolve()
                              for p in (args.video, args.weights, args.config))
    for path in (video, weights, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select a working --device')
    model = load_model(config, weights, device)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / 'detections.txt'
    info_path = output_dir / 'video_info.json'
    video_path = output_dir / 'detection.mp4'
    cap = cv2.VideoCapture(str(video))
    writer = None
    processed = detection_count = 0
    inference_seconds = 0.0
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError('Video has invalid FPS or dimensions')
        if args.save_video:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot initialize MP4 encoder: {video_path}')
        orig_size = torch.tensor([[width, height]], device=device)
        with txt_path.open('w', encoding='utf-8') as result, \
                tqdm(total=total if total > 0 else None, unit='frame') as progress, \
                torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise RuntimeError('Decoded frame dimensions differ from video metadata')
                processed += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = transforms(image).unsqueeze(0).to(device)
                labels, boxes, scores = model(input_tensor, orig_size)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - start
                frame_boxes, frame_scores = valid_detections(
                    boxes[0], scores[0], width, height, args.score_thr)
                for box, score in zip(frame_boxes, frame_scores):
                    # Round-trip float32 precision; do not round tiny boxes to zero width.
                    coords = ' '.join(f'{float(v):.9g}' for v in box)
                    result.write(f'{processed} 0 1 {float(score):.9g} {coords}\n')
                    if writer is not None:
                        x1, y1, x2, y2 = np.rint(box).astype(int)
                        x1, x2 = np.clip([x1, x2], 0, width - 1)
                        y1, y2 = np.clip([y1, y2], 0, height - 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(frame, f'{score:.3f}', (x1, max(12, y1 - 3)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255),
                                    1, cv2.LINE_AA)
                detection_count += len(frame_boxes)
                if writer is not None:
                    writer.write(frame)
                del input_tensor, labels, boxes, scores
                progress.update(1)
        if processed == 0:
            raise RuntimeError('No video frames could be decoded')
        if total > 0 and processed != total:
            raise RuntimeError(f'Incomplete decode: expected {total}, processed {processed}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    info = dict(video_path=str(video), fps=fps, width=width, height=height,
                total_frames=total, processed_frames=processed, weights_path=str(weights),
                config_path=str(config), score_threshold=args.score_thr,
                detection_count=detection_count, save_video=args.save_video,
                average_inference_seconds=inference_seconds / processed,
                inference_timing='RGB conversion, preprocessing, device transfer and model/postprocessor; CUDA synchronized')
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Total frames: {total}; processed: {processed}; detections: {detection_count}')
    print(f'Average detections/frame: {detection_count / processed:.3f}')
    print(f'Average inference time/frame: {inference_seconds / processed * 1000:.3f} ms')
    print(f'Detections: {txt_path}\nMetadata: {info_path}')
    if args.save_video:
        print(f'Visualization: {video_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default=str(ROOT / 'MultiUAV-002.mp4'))
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--config', default=str(ROOT / 'configs/gqdetr/gqdetr_r50vd_6x_coco.yml'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/MultiUAV-002'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--save-video', action=argparse.BooleanOptionalAction, default=True,
                        help='Save visualization (default); use --no-save-video to disable')
    main(parser.parse_args())
