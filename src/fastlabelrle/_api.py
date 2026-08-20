from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _core

_SUPPORTED_DTYPES = (np.dtype(np.uint32), np.dtype(np.uint64))


class EncodedLabels(NamedTuple):
    """COCO compressed RLE counts keyed by their original instance IDs."""

    ids: np.ndarray
    counts: list[bytes]


def encode(labels: np.ndarray) -> EncodedLabels:
    """Encode all nonzero instances in a 2D integer label image as COCO RLE.

    The input is scanned directly, so no full-resolution binary mask is materialized
    per instance. Returned IDs are sorted in ascending order and preserve sparse IDs.
    ``counts`` contains the standard compressed COCO RLE byte strings used by
    ``pycocotools``. The RLE size is ``labels.shape``.
    """
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array")
    if labels.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("labels dtype must be uint32 or uint64")
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    ids, counts = _core.encode(np.ascontiguousarray(labels))
    return EncodedLabels(ids, counts)
