# fastlabelrle

Fast direct COCO RLE encoding for integer instance-label images.

Most COCO tooling encodes binary masks. If a segmentation already exists as a single integer
label image, a common workflow is therefore:

```python
for instance_id in ids:
    binary = labels == instance_id
    rle = encode(binary)
```

That repeatedly scans the full image and materializes one full-resolution binary mask per
instance. `fastlabelrle` instead scans the integer label image directly and emits compressed
COCO RLE counts for every nonzero label.

## Usage

```python
import numpy as np
from fastlabelrle import encode

labels = np.array(
    [
        [0, 0, 17, 17],
        [0, 91, 91, 17],
        [0, 91, 0, 5002],
    ],
    dtype=np.uint32,
)

encoded = encode(labels)
print(encoded.ids)     # [  17   91 5002]
print(encoded.counts)  # compressed COCO RLE bytes, one entry per ID
```

The original sparse label IDs are preserved. `uint32` and `uint64` label images are supported.
The RLE image size is `labels.shape`.

To construct standard COCO segmentation dictionaries:

```python
rles = [
    {"size": list(labels.shape), "counts": counts}
    for counts in encoded.counts
]
```

## Why it is fast

The speedup comes from avoiding the binary-mask abstraction entirely. `fastlabelrle` scans the
integer label image once in COCO column-major order, collects foreground runs for each observed
label, and converts those runs directly to compressed COCO counts.

The comparison libraries below are already fast binary-mask encoders. The expensive part in this
use case is constructing one full-resolution binary mask per instance before those encoders can
run.

## Benchmarks

These are end-to-end timings from the same `uint32` label image to COCO RLEs for every instance.
They include binary-mask construction for libraries whose APIs require binary masks. They are not
encoder-only benchmarks.

Environment: macOS, CPython 3.12.13, NumPy 2.5.2, pycocotools 2.0.11,
rpycocotools 0.0.7, hotcoco 0.5.0.

| Label image | Instances | fastlabelrle | pycocotools | rpycocotools | hotcoco |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 x 1024 | 1,024 | **3.01 ms** | 4.57 s | 2.40 s | 5.08 s |
| 2048 x 2048 | 4,096 | **35.72 ms** | 84.92 s | 49.06 s | 95.33 s |

On the 2048 x 2048 / 4,096-instance case, that is approximately **2,378x** faster than the
pycocotools workflow, **1,374x** faster than rpycocotools, and **2,669x** faster than hotcoco.

Batching does not remove the representation cost. With batch size 32 on the same 2048 x 2048
case, pycocotools took 84.55 s and hotcoco took 106.89 s, while each temporary binary batch was
128 MiB. The original `uint32` label image was 16 MiB.

The benchmark uses deterministic non-overlapping ellipse-like instances. Results will vary with
image size, instance count, mask fragmentation, hardware, and library versions.

Run it yourself:

```bash
uv sync --group benchmark
uv run python benchmarks/benchmark.py --size 1024 --instances 1024 --repeats 5
uv run python benchmarks/benchmark.py --size 2048 --instances 4096 --repeats 1 --batch-sizes 32
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Scope

`fastlabelrle` intentionally has one job: direct 2D integer-label-image to compressed COCO RLE
encoding on CPU. It has no dependency on `fastlabelops` and does not relabel or otherwise modify
instance IDs.
