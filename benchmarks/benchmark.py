from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import fastlabelrle


def make_nuclei_mask(size: int, n_instances: int, seed: int = 0) -> np.ndarray:
    """Create deterministic non-overlapping ellipse-like instances."""
    if n_instances < 1:
        raise ValueError("n_instances must be positive")

    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.uint32)
    cols = math.ceil(math.sqrt(n_instances))
    rows = math.ceil(n_instances / cols)
    cell_h = size / rows
    cell_w = size / cols

    for index in range(n_instances):
        row, col = divmod(index, cols)
        cy = (row + 0.5) * cell_h
        cx = (col + 0.5) * cell_w
        ry = max(1.0, cell_h * rng.uniform(0.20, 0.34))
        rx = max(1.0, cell_w * rng.uniform(0.20, 0.34))

        y0 = max(0, int(cy - ry - 1))
        y1 = min(size, int(cy + ry + 2))
        x0 = max(0, int(cx - rx - 1))
        x1 = min(size, int(cx + rx + 2))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        patch = labels[y0:y1, x0:x1]
        patch[inside] = index + 1

    return labels


def time_call(fn: Callable[[], Any], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("benchmark function unexpectedly returned None")
        samples.append(elapsed)
    return statistics.median(samples)


def bench_fastlabelrle(labels: np.ndarray) -> Any:
    return fastlabelrle.encode(labels)


def make_pycocotools(labels: np.ndarray, ids: np.ndarray) -> Callable[[], Any] | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[dict[str, Any]]:
        out = []
        for label in ids:
            binary = np.asfortranarray(labels == label, dtype=np.uint8)
            out.append(mask_utils.encode(binary))
        return out

    return run


def make_rpycocotools(labels: np.ndarray, ids: np.ndarray) -> Callable[[], Any] | None:
    try:
        from rpycocotools import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[Any]:
        out = []
        for label in ids:
            binary = np.asarray(labels == label, dtype=np.uint8)
            out.append(mask_utils.encode(binary, target="coco_rle"))
        return out

    return run


def make_hotcoco(labels: np.ndarray, ids: np.ndarray) -> Callable[[], Any] | None:
    try:
        from hotcoco import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[dict[str, Any]]:
        out = []
        for label in ids:
            binary = np.asarray(labels == label, dtype=np.uint8)
            out.append(mask_utils.encode(binary))
        return out

    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end integer-label-image -> COCO RLE benchmark."
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--instances", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    labels = make_nuclei_mask(args.size, args.instances)
    ids = np.unique(labels)
    ids = ids[ids != 0]

    methods: list[tuple[str, Callable[[], Any] | None]] = [
        ("fastlabelrle", lambda: bench_fastlabelrle(labels)),
        ("pycocotools", make_pycocotools(labels, ids)),
        ("rpycocotools", make_rpycocotools(labels, ids)),
        ("hotcoco", make_hotcoco(labels, ids)),
    ]

    print(f"shape: {labels.shape[0]}x{labels.shape[1]}")
    print(f"instances: {len(ids)}")
    print(f"input: {labels.nbytes / 1024**2:.2f} MiB uint32")
    print()

    timings: dict[str, float] = {}
    for name, fn in methods:
        if fn is None:
            print(f"{name:14s} not installed")
            continue
        seconds = time_call(fn, args.repeats)
        timings[name] = seconds
        print(f"{name:14s} {seconds:10.6f} s")

    direct = timings.get("fastlabelrle")
    if direct is not None:
        print()
        for name, seconds in timings.items():
            if name == "fastlabelrle":
                continue
            print(f"speedup vs {name:12s}: {seconds / direct:8.1f}x")


if __name__ == "__main__":
    main()
