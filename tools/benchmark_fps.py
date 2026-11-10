"""
Benchmark RT-DETR inference speed.

Recommended execution directory:
    RT-DETR/rtdetr_pytorch

Examples:
    # Pure model inference with a synthetic input
    python tools/benchmark_fps.py \
        -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        -r output/rtdetr_r50vd/best.pth \
        --source dummy \
        --input-size 640 640 \
        --warmup 100 \
        --iterations 1000

    # Use one real image
    python tools/benchmark_fps.py \
        -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        -r output/rtdetr_r50vd/best.pth \
        --source image \
        --image-path test.jpg \
        --input-size 640 640

    # Read images from cfg.val_dataloader
    python tools/benchmark_fps.py \
        -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        -r output/rtdetr_r50vd/best.pth \
        --source dataset \
        --max-images 1000
"""

import os
import sys
import argparse
import platform
from typing import Any, Dict, Tuple

# Add rtdetr_pytorch to Python path.
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark RT-DETR PyTorch inference FPS."
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the RT-DETR YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        required=True,
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="dummy",
        choices=["dummy", "image", "dataset"],
        help=(
            "Input source. "
            "'dummy' is recommended for pure inference-speed comparison."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Image path when --source image is used.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("HEIGHT", "WIDTH"),
        help="Network input size for dummy/image mode.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Use batch size 1 for latency/FPS comparison.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warm-up iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of timed iterations in dummy/image mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of timed images in dataset mode.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Inference precision.",
    )
    parser.add_argument(
        "--include-postprocessor",
        action="store_true",
        help="Include RT-DETR postprocessor in the timed region.",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Do not call model.deploy().",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile when supported.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last memory format.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Read an RT-DETR checkpoint.

    Common RT-DETR checkpoint formats:
        checkpoint["ema"]["module"]
        checkpoint["model"]
        direct state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        ema_state = checkpoint["ema"]

        if isinstance(ema_state, dict) and "module" in ema_state:
            return ema_state["module"]

        if isinstance(ema_state, dict):
            return ema_state

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        # The checkpoint itself may already be a state_dict.
        return checkpoint

    raise TypeError(
        f"Unsupported checkpoint format: {type(checkpoint)}"
    )


def build_model(
    config_path: str,
    checkpoint_path: str,
    use_deploy: bool = True,
) -> Tuple[YAMLConfig, nn.Module]:
    """
    Construct RT-DETR from YAML and restore checkpoint weights.
    """
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    state_dict = load_checkpoint_state(checkpoint_path)

    load_result = cfg.model.load_state_dict(state_dict, strict=True)

    print("Checkpoint loaded successfully.")
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

    model = cfg.model

    if use_deploy:
        if hasattr(model, "deploy"):
            model = model.deploy()
            print("Converted model to deploy mode.")
        else:
            print(
                "Warning: model has no deploy() method; "
                "the original eval model is used."
            )

    return cfg, model


class InferenceWrapper(nn.Module):
    """
    Optional wrapper for measuring model + postprocessor.

    RT-DETR postprocessor usually receives:
        outputs
        orig_target_sizes
    """

    def __init__(
        self,
        model: nn.Module,
        postprocessor: nn.Module = None,
        include_postprocessor: bool = False,
    ):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.include_postprocessor = include_postprocessor

    def forward(
        self,
        images: torch.Tensor,
        orig_target_sizes: torch.Tensor = None,
    ):
        outputs = self.model(images)

        if not self.include_postprocessor:
            return outputs

        if self.postprocessor is None:
            raise RuntimeError(
                "include_postprocessor=True, but no postprocessor was supplied."
            )

        if orig_target_sizes is None:
            height, width = images.shape[-2:]
            orig_target_sizes = torch.tensor(
                [[height, width]],
                device=images.device,
                dtype=torch.int64,
            ).repeat(images.shape[0], 1)

        return self.postprocessor(outputs, orig_target_sizes)


def prepare_precision(
    model: nn.Module,
    precision: str,
    device: torch.device,
):
    """
    Set model parameter precision.

    fp16/bf16 are only recommended on CUDA devices that support them.
    """
    if precision == "fp32":
        model = model.float()
        dtype = torch.float32

    elif precision == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 benchmarking requires a CUDA device.")
        model = model.half()
        dtype = torch.float16

    elif precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 benchmarking requires a CUDA device.")
        model = model.to(dtype=torch.bfloat16)
        dtype = torch.bfloat16

    else:
        raise ValueError(f"Unsupported precision: {precision}")

    return model, dtype


def create_dummy_input(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    channels_last: bool,
):
    images = torch.randn(
        batch_size,
        3,
        height,
        width,
        device=device,
        dtype=dtype,
    )

    if channels_last:
        images = images.contiguous(
            memory_format=torch.channels_last
        )

    orig_sizes = torch.tensor(
        [[height, width]],
        device=device,
        dtype=torch.int64,
    ).repeat(batch_size, 1)

    return images, orig_sizes


def create_image_input(
    image_path: str,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    channels_last: bool,
):
    if image_path is None:
        raise ValueError(
            "--image-path must be supplied when --source image is used."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # PIL uses (width, height).
    image = image.resize((width, height))

    tensor = TF.to_tensor(image)
    tensor = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    tensor = tensor.to(device=device, dtype=dtype)

    if channels_last:
        tensor = tensor.contiguous(
            memory_format=torch.channels_last
        )

    orig_sizes = torch.tensor(
        [[height, width]],
        device=device,
        dtype=torch.int64,
    ).repeat(batch_size, 1)

    return tensor, orig_sizes


def extract_dataset_batch(batch: Any):
    """
    Extract images and targets from an RT-DETR data-loader batch.

    The standard collate function generally returns:
        images, targets

    images may be:
        Tensor [B, C, H, W]
        list[Tensor]
        nested tensor-like object
    """
    if not isinstance(batch, (list, tuple)) or len(batch) < 1:
        raise TypeError(
            "Expected the dataloader to return a tuple/list containing images."
        )

    images = batch[0]
    targets = batch[1] if len(batch) > 1 else None

    if isinstance(images, (list, tuple)):
        images = torch.stack(images, dim=0)

    # Compatibility with nested tensor-like implementations.
    if not torch.is_tensor(images):
        if hasattr(images, "tensors"):
            images = images.tensors
        else:
            raise TypeError(
                f"Unsupported image batch type: {type(images)}"
            )

    return images, targets


def get_original_sizes(
    targets: Any,
    images: torch.Tensor,
    device: torch.device,
):
    """
    Obtain [height, width] for each image.

    RT-DETR COCO targets commonly contain orig_size.
    The function falls back to the tensor input size.
    """
    batch_size = images.shape[0]
    input_height, input_width = images.shape[-2:]

    if isinstance(targets, (list, tuple)) and len(targets) == batch_size:
        size_list = []

        for target in targets:
            if isinstance(target, dict):
                if "orig_size" in target:
                    size = target["orig_size"]
                elif "size" in target:
                    size = target["size"]
                else:
                    size = None

                if size is not None:
                    if torch.is_tensor(size):
                        size = size.detach().view(-1).tolist()

                    if len(size) >= 2:
                        size_list.append([int(size[0]), int(size[1])])
                        continue

            size_list.append([input_height, input_width])

        return torch.tensor(
            size_list,
            dtype=torch.int64,
            device=device,
        )

    return torch.tensor(
        [[input_height, input_width]],
        dtype=torch.int64,
        device=device,
    ).repeat(batch_size, 1)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def warmup_model(
    model: nn.Module,
    images: torch.Tensor,
    orig_sizes: torch.Tensor,
    warmup_iterations: int,
    device: torch.device,
):
    for _ in range(warmup_iterations):
        _ = model(images, orig_sizes)

    synchronize(device)


@torch.inference_mode()
def benchmark_fixed_input(
    model: nn.Module,
    images: torch.Tensor,
    orig_sizes: torch.Tensor,
    warmup_iterations: int,
    iterations: int,
    device: torch.device,
):
    """
    Benchmark repeated inference on an already prepared GPU tensor.

    This excludes:
        disk I/O
        CPU decoding
        preprocessing
        CPU-to-GPU transfer
    """
    warmup_model(
        model=model,
        images=images,
        orig_sizes=orig_sizes,
        warmup_iterations=warmup_iterations,
        device=device,
    )

    batch_size = images.shape[0]

    if device.type == "cuda":
        iteration_times_ms = []

        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(images, orig_sizes)
            end_event.record()

            end_event.synchronize()
            iteration_times_ms.append(
                start_event.elapsed_time(end_event)
            )

        total_time_ms = float(np.sum(iteration_times_ms))

    else:
        import time

        iteration_times_ms = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model(images, orig_sizes)
            end_time = time.perf_counter()

            iteration_times_ms.append(
                (end_time - start_time) * 1000.0
            )

        total_time_ms = float(np.sum(iteration_times_ms))

    num_images = iterations * batch_size
    total_time_s = total_time_ms / 1000.0

    return {
        "num_images": num_images,
        "num_batches": iterations,
        "batch_size": batch_size,
        "total_time_s": total_time_s,
        "fps": num_images / total_time_s,
        "latency_per_batch_ms": total_time_ms / iterations,
        "latency_per_image_ms": total_time_ms / num_images,
        "std_batch_ms": float(np.std(iteration_times_ms)),
        "median_batch_ms": float(np.median(iteration_times_ms)),
        "p90_batch_ms": float(np.percentile(iteration_times_ms, 90)),
        "p95_batch_ms": float(np.percentile(iteration_times_ms, 95)),
    }


@torch.inference_mode()
def benchmark_dataset(
    model: nn.Module,
    dataloader,
    warmup_iterations: int,
    max_images: int,
    device: torch.device,
    dtype: torch.dtype,
    channels_last: bool,
):
    """
    Benchmark model inference on cfg.val_dataloader.

    Data loading and CPU-to-GPU transfer are deliberately placed outside the
    timed section. The measured result therefore remains a model-inference FPS.
    """
    if device.type != "cuda":
        raise ValueError(
            "Dataset benchmark currently expects CUDA Event timing."
        )

    warmup_count = 0
    timed_images = 0
    timed_batches = 0
    iteration_times_ms = []

    for batch in dataloader:
        images, targets = extract_dataset_batch(batch)

        images = images.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        if channels_last:
            images = images.contiguous(
                memory_format=torch.channels_last
            )

        orig_sizes = get_original_sizes(
            targets=targets,
            images=images,
            device=device,
        )

        if warmup_count < warmup_iterations:
            _ = model(images, orig_sizes)
            warmup_count += 1

            if warmup_count == warmup_iterations:
                synchronize(device)

            continue

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(images, orig_sizes)
        end_event.record()

        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times_ms.append(elapsed_ms)

        timed_images += images.shape[0]
        timed_batches += 1

        if timed_images >= max_images:
            break

    if timed_batches == 0:
        raise RuntimeError(
            "No timed batches were processed. "
            "The dataset may contain fewer batches than the warm-up count."
        )

    total_time_ms = float(np.sum(iteration_times_ms))
    total_time_s = total_time_ms / 1000.0

    return {
        "num_images": timed_images,
        "num_batches": timed_batches,
        "batch_size": "dataset-defined",
        "total_time_s": total_time_s,
        "fps": timed_images / total_time_s,
        "latency_per_batch_ms": total_time_ms / timed_batches,
        "latency_per_image_ms": total_time_ms / timed_images,
        "std_batch_ms": float(np.std(iteration_times_ms)),
        "median_batch_ms": float(np.median(iteration_times_ms)),
        "p90_batch_ms": float(np.percentile(iteration_times_ms, 90)),
        "p95_batch_ms": float(np.percentile(iteration_times_ms, 95)),
    }


def print_environment(
    args,
    device: torch.device,
    model: nn.Module,
):
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("\n" + "=" * 72)
    print("RT-DETR inference benchmark")
    print("=" * 72)
    print(f"Config              : {args.config}")
    print(f"Checkpoint          : {args.resume}")
    print(f"Source              : {args.source}")
    print(f"Device              : {device}")
    print(f"Precision           : {args.precision}")
    print(f"Deploy mode         : {not args.no_deploy}")
    print(f"Include postprocess : {args.include_postprocessor}")
    print(f"torch.compile       : {args.compile}")
    print(f"Channels last       : {args.channels_last}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"Python              : {platform.python_version()}")
    print(f"Parameters          : {parameter_count / 1e6:.3f} M")

    if device.type == "cuda":
        print(f"CUDA                : {torch.version.cuda}")
        print(
            f"GPU                 : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print("=" * 72)


def print_results(results: Dict[str, Any]):
    print("\n" + "=" * 72)
    print("Benchmark results")
    print("=" * 72)
    print(f"Timed images        : {results['num_images']}")
    print(f"Timed batches       : {results['num_batches']}")
    print(f"Batch size          : {results['batch_size']}")
    print(f"Total GPU time      : {results['total_time_s']:.6f} s")
    print(f"FPS                 : {results['fps']:.3f}")
    print(
        f"Latency / image     : "
        f"{results['latency_per_image_ms']:.3f} ms"
    )
    print(
        f"Latency / batch     : "
        f"{results['latency_per_batch_ms']:.3f} ms"
    )
    print(
        f"Median batch latency: "
        f"{results['median_batch_ms']:.3f} ms"
    )
    print(
        f"P90 batch latency   : "
        f"{results['p90_batch_ms']:.3f} ms"
    )
    print(
        f"P95 batch latency   : "
        f"{results['p95_batch_ms']:.3f} ms"
    )
    print(
        f"Batch latency std   : "
        f"{results['std_batch_ms']:.3f} ms"
    )
    print("=" * 72)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    # Fixed-size convolution benchmarking is generally faster with benchmark.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    cfg, detector = build_model(
        config_path=args.config,
        checkpoint_path=args.resume,
        use_deploy=not args.no_deploy,
    )

    postprocessor = None

    if args.include_postprocessor:
        postprocessor = cfg.postprocessor

        if hasattr(postprocessor, "deploy"):
            postprocessor = postprocessor.deploy()

    detector.eval()
    detector.to(device)

    detector, input_dtype = prepare_precision(
        model=detector,
        precision=args.precision,
        device=device,
    )

    if args.channels_last:
        detector = detector.to(
            memory_format=torch.channels_last
        )

    model = InferenceWrapper(
        model=detector,
        postprocessor=postprocessor,
        include_postprocessor=args.include_postprocessor,
    )

    model.eval()
    model.to(device)

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "The installed PyTorch version has no torch.compile."
            )

        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
        )

    print_environment(args, device, model)

    height, width = args.input_size

    if args.source == "dummy":
        images, orig_sizes = create_dummy_input(
            batch_size=args.batch_size,
            height=height,
            width=width,
            device=device,
            dtype=input_dtype,
            channels_last=args.channels_last,
        )

        results = benchmark_fixed_input(
            model=model,
            images=images,
            orig_sizes=orig_sizes,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            device=device,
        )

    elif args.source == "image":
        images, orig_sizes = create_image_input(
            image_path=args.image_path,
            batch_size=args.batch_size,
            height=height,
            width=width,
            device=device,
            dtype=input_dtype,
            channels_last=args.channels_last,
        )

        results = benchmark_fixed_input(
            model=model,
            images=images,
            orig_sizes=orig_sizes,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            device=device,
        )

    elif args.source == "dataset":
        if not hasattr(cfg, "val_dataloader"):
            raise AttributeError(
                "YAMLConfig has no val_dataloader. "
                "Check whether the YAML configuration defines val_dataloader."
            )

        dataloader = cfg.val_dataloader

        results = benchmark_dataset(
            model=model,
            dataloader=dataloader,
            warmup_iterations=args.warmup,
            max_images=args.max_images,
            device=device,
            dtype=input_dtype,
            channels_last=args.channels_last,
        )

    else:
        raise ValueError(f"Unsupported source: {args.source}")

    print_results(results)


if __name__ == "__main__":
    main()
    
    



# CUDA_VISIBLE_DEVICES=0 python tools/benchmark_fps.py \
#     -c configs/gqdetr/gqdetr_light_6x_coco.yml \
#     -r exps_light/20260525_223959_r18_backbone_woEN_FCM/last.pth \
#     --source dummy \
#     --input-size 640 640 \
#     --batch-size 1 \
#     --warmup 100 \
#     --iterations 1000 \
#     --precision fp32 \
#     --device cuda:0