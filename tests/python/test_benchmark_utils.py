"""Unit tests for saq.benchmark module — no C++ required."""

from __future__ import annotations

import numpy as np
import pytest

from saq.benchmark import compute_ground_truth, recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        results = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        gt = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        assert recall_at_k(results, gt, k=3) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        results = np.array([[10, 11, 12]], dtype=np.int32)
        gt = np.array([[0, 1, 2]], dtype=np.int32)
        assert recall_at_k(results, gt, k=3) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        results = np.array([[0, 99, 98]], dtype=np.int32)
        gt = np.array([[0, 1, 2]], dtype=np.int32)
        assert recall_at_k(results, gt, k=2) == pytest.approx(0.5)

    def test_k_larger_than_results(self) -> None:
        results = np.array([[0, 1]], dtype=np.int32)
        gt = np.array([[0, 1, 2, 3]], dtype=np.int32)
        r = recall_at_k(results, gt, k=4)
        assert 0.0 <= r <= 1.0

    def test_batch_averaging(self) -> None:
        results = np.array([[0, 1], [9, 8]], dtype=np.int32)
        gt = np.array([[0, 1], [0, 1]], dtype=np.int32)
        assert recall_at_k(results, gt, k=2) == pytest.approx(0.5)


class TestComputeGroundTruth:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 8)).astype(np.float32)
        queries = rng.standard_normal((10, 8)).astype(np.float32)
        gt = compute_ground_truth(base, queries, top_k=5)
        assert gt.shape == (10, 5)

    def test_nearest_neighbor_correctness(self) -> None:
        base = np.eye(10, 10, dtype=np.float32)
        query = np.zeros((1, 10), dtype=np.float32)
        query[0, 7] = 1.0
        gt = compute_ground_truth(base, query, top_k=1)
        assert gt[0, 0] == 7

    def test_no_duplicate_ids(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.standard_normal((50, 16)).astype(np.float32)
        queries = rng.standard_normal((5, 16)).astype(np.float32)
        gt = compute_ground_truth(base, queries, top_k=10)
        for row in gt:
            assert len(set(row.tolist())) == len(row), "duplicate IDs in ground truth"
