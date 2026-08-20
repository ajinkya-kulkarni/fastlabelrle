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


def _make_fortran_binary(labels: np.ndarray, label: np.integer[Any]) -> np.ndarray:
    """Build one uint8 binary mask directly in Fortran order."""
    binary = np.empty(labels.shape, dtype=np.uint8, order="F")
    np.equal(labels, label, out=binary)
    return binary


def _make_fortran_batch(labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Build an HxWxN uint8 mask stack directly in Fortran order."""
    batch = np.empty((*labels.shape, len(ids)), dtype=np.uint8, order="F")
    for index, label in enumerate(ids):
        np.equal(labels, label, out=batch[:, :, index])
    return batch


def _encode_in_batches(
    labels: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
    encode: Callable[[np.ndarray], Any],
) -> list[Any]:
    out: list[Any] = []
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start : start + batch_size]
        binary = _make_fortran_batch(labels, chunk_ids)
        encoded = encode(binary)
        if isinstance(encoded, list):
            out.extend(encoded)
        else:
            out.append(encoded)
    return out


def make_pycocotools(labels: np.ndarray, ids: np.ndarray) -> Callable[[], Any] | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[dict[str, Any]]:
        out = []
        for label in ids:
            out.append(mask_utils.encode(_make_fortran_binary(labels, label)))
        return out

    return run


def make_pycocotools_batched(
    labels: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
) -> Callable[[], Any] | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[Any]:
        return _encode_in_batches(labels, ids, batch_size, mask_utils.encode)

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
            out.append(mask_utils.encode(_make_fortran_binary(labels, label)))
        return out

    return run


def make_hotcoco_batched(
    labels: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
) -> Callable[[], Any] | None:
    try:
        from hotcoco import mask as mask_utils
    except ImportError:
        return None

    def run() -> list[Any]:
        return _encode_in_batches(labels, ids, batch_size, mask_utils.encode)

    return run


def _parse_batch_sizes(value: str) -> list[int]:
    if not value:
        return []

    sizes: list[int] = []
    for item in value.split(","):
        size = int(item.strip())
        if size < 1:
            raise argparse.ArgumentTypeError("batch sizes must be positive")
        sizes.append(size)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end integer-label-image -> COCO RLE benchmark."
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--instances", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=[8, 32, 128],
        help="comma-separated binary-mask batch sizes for pycocotools/hotcoco",
    )
    parser.add_argument(
        "--max-batch-mib",
        type=float,
        default=512.0,
        help="skip a batched baseline if its temporary HxWxN uint8 stack exceeds this size",
    )
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

    batch_notes: dict[str, float] = {}
    for batch_size in args.batch_sizes:
        batch_mib = labels.shape[0] * labels.shape[1] * batch_size / 1024**2
        if batch_mib > args.max_batch_mib:
            continue

        py_name = f"pycoco[b={batch_size}]"
        hot_name = f"hotcoco[b={batch_size}]"
        methods.append((py_name, make_pycocotools_batched(labels, ids, batch_size)))
        methods.append((hot_name, make_hotcoco_batched(labels, ids, batch_size)))
        batch_notes[py_name] = batch_mib
        batch_notes[hot_name] = batch_mib

    print(f"shape: {labels.shape[0]}x{labels.shape[1]}")
    print(f"instances: {len(ids)}")
    print(f"input: {labels.nbytes / 1024**2:.2f} MiB uint32")
    if batch_notes:
        max_batch = max(batch_notes.values())
        print(f"largest temporary binary batch: {max_batch:.2f} MiB")
    print()

    timings: dict[str, float] = {}
    for name, fn in methods:
        if fn is None:
            print(f"{name:18s} not installed")
            continue
        seconds = time_call(fn, args.repeats)
        timings[name] = seconds
        batch_mib = batch_notes.get(name)
        note = "" if batch_mib is None else f"  ({batch_mib:.1f} MiB binary batch)"
        print(f"{name:18s} {seconds:10.6f} s{note}")

    direct = timings.get("fastlabelrle")
    if direct is not None:
        print()
        for name, seconds in timings.items():
            if name == "fastlabelrle":
                continue
            print(f"speedup vs {name:16s}: {seconds / direct:8.1f}x")


if __name__ == "__main__":
    main()
