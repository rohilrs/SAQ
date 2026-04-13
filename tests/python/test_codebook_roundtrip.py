#!/usr/bin/env python3
"""Lightweight codebook encode/search round-trip test.

Uses synthetic data only — no DBPedia, no GPU required.
Tests that Gaussian codebook quantization produces reasonable recall
compared to uniform quantization on the same data.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from saq._saq_core import IVF, QuantizeConfig, SearcherConfig


def make_gaussian_codebooks(bits_list: list[int]) -> dict[int, np.ndarray]:
    """Generate Gaussian quantile-based base codebooks for each bit width."""
    cb = {}
    for bits in bits_list:
        k = 1 << bits
        quantiles = np.array(
            [norm.ppf((i + 0.5) / k) for i in range(k)], dtype=np.float32
        )
        cb[bits] = quantiles
    return cb


def compute_recall(results: np.ndarray, gt: np.ndarray, k: int) -> float:
    nq = results.shape[0]
    recall = 0.0
    for q in range(nq):
        gt_set = set(int(x) for x in gt[q, :k])
        found = set(int(x) for x in results[q, :k])
        recall += len(gt_set & found) / k
    return recall / nq


def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact L2 k-NN via brute force."""
    # data: (N, D), queries: (nq, D)
    dists = np.sum((queries[:, None, :] - data[None, :, :]) ** 2, axis=2)  # (nq, N)
    return np.argsort(dists, axis=1)[:, :k].astype(np.int32)


def test_codebook_vs_uniform():
    """Codebook quantization should produce non-trivial recall on synthetic data."""
    rng = np.random.default_rng(123)
    N, D, K = 2000, 128, 32
    BPD = 4.0
    NPROBE = 16
    TOPK = 10

    # Synthetic clustered data
    centers = rng.standard_normal((K, D)).astype(np.float32) * 3
    labels = rng.integers(0, K, size=N).astype(np.uint32)
    X = np.empty((N, D), dtype=np.float32)
    for i in range(N):
        X[i] = centers[labels[i]] + rng.standard_normal(D).astype(np.float32) * 0.5
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(K)], dtype=np.float32)

    nq = 50
    Q = np.empty((nq, D), dtype=np.float32)
    for i in range(nq):
        c = rng.integers(0, K)
        Q[i] = centers[c] + rng.standard_normal(D).astype(np.float32) * 0.5

    gt = brute_force_knn(X, Q, TOPK)
    var = X.var(axis=0)
    codebooks = make_gaussian_codebooks([4])
    scfg = SearcherConfig()

    # Uniform quantization
    cfg_u = QuantizeConfig()
    cfg_u.avg_bits = BPD
    cfg_u.single.random_rotation = True
    cfg_u.enable_segmentation = True
    ivf_u = IVF(N, D, K, cfg_u)
    ivf_u.set_variance(var)
    ivf_u.construct(X, centroids, labels, 1)
    r_u = ivf_u.search_batch(Q, TOPK, NPROBE, scfg)
    recall_u = compute_recall(r_u, gt, TOPK)

    # Codebook quantization
    cfg_c = QuantizeConfig()
    cfg_c.avg_bits = BPD
    cfg_c.single.random_rotation = True
    cfg_c.enable_segmentation = True
    ivf_c = IVF(N, D, K, cfg_c)
    ivf_c.set_variance(var)
    ivf_c.set_gaussian_codebooks(codebooks, var)
    assert ivf_c.has_codebooks
    ivf_c.construct(X, centroids, labels, 1)
    r_c = ivf_c.search_batch(Q, TOPK, NPROBE, scfg)
    recall_c = compute_recall(r_c, gt, TOPK)

    print(f"Uniform  R@{TOPK} = {recall_u:.4f}")
    print(f"Codebook R@{TOPK} = {recall_c:.4f}")

    # Both should achieve non-trivial recall on clustered data
    assert recall_u > 0.3, f"Uniform recall too low: {recall_u:.4f}"
    assert recall_c > 0.3, f"Codebook recall too low: {recall_c:.4f}"

    # Codebook recall should be within reasonable range of uniform
    assert recall_c > recall_u * 0.5, (
        f"Codebook recall ({recall_c:.4f}) much worse than uniform ({recall_u:.4f})"
    )


def test_codebook_save_load():
    """Codebook index should produce identical results after save/load."""
    rng = np.random.default_rng(456)
    N, D, K = 500, 64, 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    centroids = rng.standard_normal((K, D)).astype(np.float32)
    labels = rng.integers(0, K, size=N).astype(np.uint32)
    Q = rng.standard_normal((5, D)).astype(np.float32)
    var = X.var(axis=0)

    cfg = QuantizeConfig()
    cfg.avg_bits = 4.0
    cfg.single.random_rotation = True
    cfg.enable_segmentation = True

    codebooks = make_gaussian_codebooks([4])

    ivf = IVF(N, D, K, cfg)
    ivf.set_variance(var)
    ivf.set_gaussian_codebooks(codebooks, var)
    ivf.construct(X, centroids, labels, 1)

    scfg = SearcherConfig()
    r1 = ivf.search_batch(Q, 3, 4, scfg)

    path = "/tmp/test_codebook_roundtrip.idx"
    ivf.save(path)

    ivf2 = IVF()
    ivf2.load(path)
    r2 = ivf2.search_batch(Q, 3, 4, scfg)

    assert np.array_equal(r1, r2), "Results differ after save/load"
    print("Save/load round-trip: results match")


def test_invalid_codebook_rejected():
    """Codebook with wrong entry count should raise."""
    cfg = QuantizeConfig()
    cfg.avg_bits = 4.0
    ivf = IVF(100, 64, 4, cfg)
    var = np.ones(64, dtype=np.float32)

    bad_cb = {4: np.random.randn(10).astype(np.float32)}  # 10 != 2^4 = 16
    try:
        ivf.set_gaussian_codebooks(bad_cb, var)
        assert False, "Should have raised"
    except Exception as e:
        assert "expected 16" in str(e)
        print(f"Invalid codebook correctly rejected: {e}")


if __name__ == "__main__":
    test_codebook_vs_uniform()
    test_codebook_save_load()
    test_invalid_codebook_rejected()
    print("\nAll codebook round-trip tests passed.")
