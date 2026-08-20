from __future__ import annotations

import numpy as np
import pytest

from fastlabelrle import encode


def _compress_counts(counts: list[int]) -> bytes:
    out = bytearray()
    for i, count in enumerate(counts):
        x = count
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if c & 0x10 else x != 0
            if more:
                c |= 0x20
            out.append(c + 48)
    return bytes(out)


def _reference_counts(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    previous = 0
    run = 0
    counts: list[int] = []
    for value in flat:
        current = int(value != 0)
        if current != previous:
            counts.append(run)
            run = 0
            previous = current
        run += 1
    counts.append(run)
    return counts


def _reference(labels: np.ndarray, label: int) -> bytes:
    return _compress_counts(_reference_counts(labels == label))


def test_known_sparse_labels() -> None:
    labels = np.array(
        [
            [0, 0, 17, 17],
            [0, 91, 91, 17],
            [0, 91, 0, 5002],
            [0, 0, 5002, 5002],
        ],
        dtype=np.uint32,
    )

    result = encode(labels)

    assert result.ids.tolist() == [17, 91, 5002]
    assert result.counts == [_reference(labels, int(label)) for label in result.ids]


def test_random_exact_reference_parity() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        labels = rng.integers(0, 18, size=(31, 37), dtype=np.uint32)
        result = encode(labels)
        assert result.ids.tolist() == sorted(set(labels.ravel().tolist()) - {0})
        for label, counts in zip(result.ids, result.counts, strict=True):
            assert counts == _reference(labels, int(label))


def test_uint64_and_sparse_ids() -> None:
    labels = np.array([[0, 2**40], [7, 2**40]], dtype=np.uint64)
    result = encode(labels)
    assert result.ids.dtype == np.uint64
    assert result.ids.tolist() == [7, 2**40]
    for label, counts in zip(result.ids, result.counts, strict=True):
        assert counts == _reference(labels, int(label))


def test_noncontiguous_input_is_supported() -> None:
    labels = np.arange(36, dtype=np.uint32).reshape(6, 6)[:, ::2]
    assert not labels.flags.c_contiguous
    result = encode(labels)
    assert result.ids.size == np.unique(labels[labels != 0]).size


def test_empty_background_only_image() -> None:
    labels = np.zeros((8, 9), dtype=np.uint32)
    result = encode(labels)
    assert result.ids.size == 0
    assert result.counts == []


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError, match="uint32 or uint64"):
        encode(np.zeros((3, 3), dtype=np.int32))


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        encode(np.zeros((2, 3, 4), dtype=np.uint32))
