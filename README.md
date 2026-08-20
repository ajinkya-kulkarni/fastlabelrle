# fastlabelrle

Fast direct COCO RLE encoding for integer instance-label images.

Most COCO tooling encodes one binary mask at a time. If a segmentation already exists as a
single integer label image, a common workflow is therefore:

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
The RLE image size is simply `labels.shape`.

To construct standard COCO segmentation dictionaries:

```python
rles = [
    {"size": list(labels.shape), "counts": counts}
    for counts in encoded.counts
]
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Native benchmark

Install the comparison libraries:

```bash
uv sync --group benchmark
```

Then benchmark the full workflow from the same integer label image:

```bash
uv run python benchmarks/benchmark.py --size 1024 --instances 1024 --repeats 3
```

The benchmark compares against `pycocotools`, `rpycocotools`, and `hotcoco`, including the
cost of creating the per-instance binary masks those APIs require. That end-to-end comparison
is the intended use case; encoder-only timings answer a different question.

## Status

Early prototype. The API and implementation may change while native-library benchmarks and
real segmentation masks are being validated.
