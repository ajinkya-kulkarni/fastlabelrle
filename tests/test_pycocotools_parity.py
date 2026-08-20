from __future__ import annotations

import numpy as np
import pytest

from fastlabelrle import encode

mask_utils = pytest.importorskip("pycocotools.mask")


def test_compressed_bytes_match_pycocotools() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 32, size=(79, 83), dtype=np.uint32)
    result = encode(labels)

    for label, counts in zip(result.ids, result.counts, strict=True):
        binary = np.asfortranarray(labels == label, dtype=np.uint8)
        expected = mask_utils.encode(binary)["counts"]
        assert counts == expected
