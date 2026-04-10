"""Shared pytest fixtures for SAQ Python tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make python/ directory importable so `from preprocessing.utils.io import ...` works
_python_dir = Path(__file__).parent.parent.parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))


SYNTH_N = 1000
SYNTH_DIM = 64
SYNTH_NQ = 50
SYNTH_K = 32
SYNTH_SEED = 42


@pytest.fixture(scope="session")
def synth_rng() -> np.random.Generator:
    return np.random.default_rng(SYNTH_SEED)


@pytest.fixture(scope="session")
def synth_base(synth_rng: np.random.Generator) -> np.ndarray:
    """Random float32 base vectors, shape (1000, 64)."""
    return synth_rng.standard_normal((SYNTH_N, SYNTH_DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def synth_queries(synth_rng: np.random.Generator) -> np.ndarray:
    """Random float32 query vectors, shape (50, 64)."""
    return synth_rng.standard_normal((SYNTH_NQ, SYNTH_DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def synth_ground_truth(synth_base: np.ndarray, synth_queries: np.ndarray) -> np.ndarray:
    """Exact brute-force L2 ground truth, shape (50, 100)."""
    from saq.benchmark import compute_ground_truth
    return compute_ground_truth(synth_base, synth_queries, top_k=100)


@pytest.fixture(scope="session")
def tmp_dir():
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_bindings: requires compiled _saq_core extension"
    )
    config.addinivalue_line(
        "markers", "requires_faiss: requires faiss-cpu to be installed"
    )
    config.addinivalue_line("markers", "slow: test takes >10 seconds")
