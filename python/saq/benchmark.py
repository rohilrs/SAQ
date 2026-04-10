"""Recall and benchmark utilities for SAQ search evaluation."""

from __future__ import annotations

import numpy as np


def recall_at_k(
    results: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Compute Recall@k over a batch of queries.

    Parameters
    ----------
    results:
        Integer array of shape (nq, topk) containing returned neighbor IDs.
    ground_truth:
        Integer array of shape (nq, gt_k) containing ground truth neighbor IDs.
        gt_k must be >= k.
    k:
        Recall cutoff. Counts how many of the top-k ground truth neighbors
        appear anywhere in the returned results[:k].

    Returns
    -------
    float
        Mean recall across all queries, in [0, 1].
    """
    nq = results.shape[0]
    k_res = min(k, results.shape[1])
    k_gt = min(k, ground_truth.shape[1])

    hits = 0
    total = 0
    for q in range(nq):
        gt_set = set(ground_truth[q, :k_gt].tolist())
        for idx in results[q, :k_res].tolist():
            if idx in gt_set:
                hits += 1
        total += k_gt

    return hits / total if total > 0 else 0.0


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    top_k: int = 100,
) -> np.ndarray:
    """Brute-force ground truth via exact L2 distances.

    Parameters
    ----------
    base:
        Float32 array of shape (n, d).
    queries:
        Float32 array of shape (nq, d).
    top_k:
        Number of nearest neighbors to return per query.

    Returns
    -------
    np.ndarray
        Integer array of shape (nq, top_k) containing 0-indexed neighbor IDs.
    """
    nq = queries.shape[0]
    gt = np.empty((nq, top_k), dtype=np.int32)
    for q in range(nq):
        diffs = base - queries[q]
        dists = np.einsum("nd,nd->n", diffs, diffs)
        gt[q] = np.argsort(dists)[:top_k]
    return gt
