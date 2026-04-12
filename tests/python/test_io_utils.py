"""Unit tests for python/preprocessing/utils/io.py — no C++ or faiss required."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from preprocessing.utils.io import (
    read_fvecs,
    read_ivecs,
    write_fvecs,
    write_ivecs,
)


@pytest.fixture
def tmp_path_local():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestFvecsRoundTrip:
    def test_write_then_read_float32(self, tmp_path_local: Path) -> None:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = str(tmp_path_local / "test.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_single_row(self, tmp_path_local: Path) -> None:
        data = np.array([[0.1, 0.2]], dtype=np.float32)
        path = str(tmp_path_local / "single.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        np.testing.assert_allclose(loaded, data, rtol=1e-6)

    def test_large_matrix(self, tmp_path_local: Path) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 128)).astype(np.float32)
        path = str(tmp_path_local / "large.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        assert loaded.shape == (500, 128)
        np.testing.assert_array_equal(loaded, data)

    def test_output_dtype_is_float32(self, tmp_path_local: Path) -> None:
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        path = str(tmp_path_local / "dtype.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        assert loaded.dtype == np.float32


class TestIvecsRoundTrip:
    def test_write_then_read_int32(self, tmp_path_local: Path) -> None:
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        path = str(tmp_path_local / "test.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_single_column(self, tmp_path_local: Path) -> None:
        data = np.arange(100, dtype=np.int32).reshape(100, 1)
        path = str(tmp_path_local / "single_col.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_output_dtype(self, tmp_path_local: Path) -> None:
        data = np.array([[10, 20]], dtype=np.int32)
        path = str(tmp_path_local / "dtype.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        assert loaded.dtype == np.int32
