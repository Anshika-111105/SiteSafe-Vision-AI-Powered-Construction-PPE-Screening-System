#!/usr/bin/env python3
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transforms import get_eval_transforms
from src.models.architectures import create_mobilenet_v3_model, create_resnet50_model
from src.utils.logger import setup_logger

logger = setup_logger("benchmark")


def benchmark_model_architecture(model_name: str, num_runs: int = 50) -> dict[str, Any]:
    eval_tf = get_eval_transforms(224, 256)
    dummy_img = Image.new("RGB", (300, 300), color=(120, 140, 160))
    input_tensor = eval_tf(dummy_img).unsqueeze(0)

    # 1. Measure Load Time
    t_load_0 = time.perf_counter()
    if model_name == "resnet50":
        model = create_resnet50_model(num_classes=3, pretrained=False)
        ckpt_path = PROJECT_ROOT / "artifacts" / "models" / "best_resnet50.pt"
    else:
        model = create_mobilenet_v3_model(num_classes=3, pretrained=False)
        ckpt_path = PROJECT_ROOT / "artifacts" / "models" / "best_mobilenet_v3_large.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    load_time_ms = (time.perf_counter() - t_load_0) * 1000.0

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_file_size_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else 0.0

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)

    # 2. Measure Single-Image Inference Latencies
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    # 3. Measure Batch Throughput
    batch_throughputs = {}
    for bs in [1, 4, 8, 16]:
        batch_tensor = torch.cat([input_tensor] * bs, dim=0)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                _ = model(batch_tensor)
        dur = time.perf_counter() - t0
        fps = (bs * 10) / dur
        batch_throughputs[f"batch_size_{bs}"] = round(fps, 2)

    return {
        "model_name": model_name,
        "model_file_size_mb": round(model_file_size_mb, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "load_time_ms": round(load_time_ms, 2),
        "single_image_latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
        },
        "throughput_images_per_second": batch_throughputs,
    }


def run_full_benchmark() -> dict[str, Any]:
    logger.info("Executing Performance Benchmark on ResNet50 and MobileNetV3-Large...")

    resnet_bench = benchmark_model_architecture("resnet50")
    mobilenet_bench = benchmark_model_architecture("mobilenet_v3_large")

    benchmark_report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hardware": {
            "device": "CPU",
            "threads": torch.get_num_threads(),
        },
        "architectures": {
            "resnet50": resnet_bench,
            "mobilenet_v3_large": mobilenet_bench,
        },
        "comparison_summary": {
            "latency_speedup_mobilenet_vs_resnet": round(
                resnet_bench["single_image_latency_ms"]["mean"] / mobilenet_bench["single_image_latency_ms"]["mean"], 2
            ),
            "model_size_reduction_ratio": round(
                resnet_bench["model_file_size_mb"] / mobilenet_bench["model_file_size_mb"], 2
            ) if mobilenet_bench["model_file_size_mb"] > 0 else 0.0,
        },
    }

    out_file = PROJECT_ROOT / "reports" / "performance.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info(f"Performance report saved to {out_file}")
    return benchmark_report


if __name__ == "__main__":
    run_full_benchmark()
