# SAQ Implementation Notes

## Overview

This is a C++20 implementation of SAQ (Scalar Additive Quantization) from arXiv:2509.12086, aligned with the [reference repository](https://github.com/howarlii/SAQ). The codebase was refactored on 2026-02-15 to match the reference architecture, replacing the earlier custom implementation.

## Architecture Alignment with Reference

The current implementation closely follows the reference repository's design:

| Component | Reference (`saqlib`) | Our Implementation (`saq`) |
|-----------|---------------------|---------------------------|
| Quantization planning | `SaqDataMaker` | `SaqDataMaker` in `quantization_plan.h` |
| CAQ encoding | `CAQEncoder` | `CAQEncoder` in `caq_encoder.h` |
| Per-segment quantization | `QuantizerCluster` | `QuantizerCluster` in `quantizer.h` |
| Multi-segment orchestration | `SAQuantizer` | `SAQuantizer` in `saq_quantizer.h` |
| 3-stage distance estimation | `CaqCluEstimator` | `CaqCluEstimator` in `caq_estimator.h` |
| Multi-segment aggregation | `SaqCluEstimator` | `SaqCluEstimator` in `saq_estimator.h` |
| Block-level search | `SAQSearcher` | `SAQSearcher` in `saq_searcher.h` |
| Cluster data storage | `CAQClusterData` | `CAQClusterData` in `cluster_data.h` |
| Code packing | `ClusterPacker` | `ClusterPacker` in `cluster_packer.h` |
| SIMD fastscan | `fast_scan` | `fast_scan.h` |
| LUT computation | `Lut` | `Lut` in `lut.h` |
| IVF index | `IVF` | `IVF` in `ivf_index.h` |

### Key Features Matching Reference
- Per-vector, per-segment v_max scaling with `fac_rescale = |o|^2 / <o, o_a>`
- CAQ code adjustment optimization (cosine similarity maximization)
- Joint DP over 64-dim blocks for dimension segmentation + bit allocation
- Per-segment random orthogonal rotation (Householder QR)
- 3-stage search: variance pruning -> 1-bit fastscan -> accurate distance
- Short codes (MSB) + long codes (remaining bits) dual storage
- 64-bit per-segment overhead in DP budget

## MSVC Compatibility Fixes

Several fixes were needed for MSVC (Windows) compilation:

1. **`__restrict__` -> `__restrict`**: MSVC uses `__restrict` instead of GCC's `__restrict__`. Macro in `defines.h`.
2. **AlignedAllocator converting constructor**: MSVC's `_Container_proxy` requires a converting constructor in custom allocators.
3. **`_aligned_malloc`/`_aligned_free` pairing**: On MSVC, memory allocated with `_aligned_malloc` MUST be freed with `_aligned_free` (not `std::free`). Added `portable_aligned_free()` helper in `memory.h`.
4. **`align_mm` memset**: Fixed `std::memset(p, 0, size)` to `std::memset(p, 0, size * sizeof(T))` for correct byte count.
5. **1-bit quantization**: Fixed `encode_and_fac` to set `base_code.code` for `num_bits_ >= 1` (not just `> 1`), and ensured `std::move(caq.code)` happens after `caq.get_oa()` call.

## Preprocessing Pipeline

Preprocessing is done in Python using Faiss (matching reference approach):

1. **PCA** (`python/preprocessing/pca.py`): Full rotation via `faiss.PCAMatrix`, outputs `vectors_pca.fvecs`, `queries_pca.fvecs`, `centroids_{K}_pca.fvecs`, `variances_pca.fvecs`
2. **K-means** (`python/preprocessing/ivf.py`): Faiss K-means clustering, outputs `centroids_{K}.fvecs`, `cluster_ids_{K}.ivecs`
3. **Ground truth** (`python/preprocessing/compute_gt.py`): Brute-force kNN, outputs `groundtruth.ivecs`

## Benchmark Results

**Dataset:** DBpedia 100K (1536D OpenAI embeddings, K=4096 clusters)

| bpd | Compression | R@1 | R@10 | R@100 | nprobe=200 |
|-----|-------------|-----|------|-------|------------|
| 1.0 | 32x | 85.0% | 87.3% | 86.6% | Yes |
| 2.0 | 16x | 92.8% | 92.6% | 90.0% | Yes |
| 4.0 | 8x | 97.0% | 94.8% | 90.9% | Yes |

Recall increases monotonically with nprobe at all bpd levels, confirming correct algorithm behavior.

### Quantization Plan Examples

- **4 bpd**: `[0..192)@8b [192..640)@5b [640..1024)@3b [1024..1536)@2b`
- **2 bpd**: `[0..64)@8b [64..128)@5b [128..320)@3b [320..1536)@1b`
- **1 bpd**: `[0..64)@4b [64..128)@2b [128..384)@1b [384..1536)@0b`

High-variance PCA dimensions receive more bits; low-variance dimensions are coarsely quantized or omitted.
