# SAQ Source Files

Implementation files for the SAQ library. Most classes are header-only (inline in `.h`); the `.cpp` files serve as minimal translation units.

## Files

| File | Key Classes | Description |
|------|-------------|-------------|
| `quantization_plan.cpp` | SaqDataMaker | Joint DP over 64-dim blocks, variance computation, binary serialization |
| `rotator.cpp` | Rotator | Per-segment random orthogonal rotation (Householder QR) |
| `caq_encoder.cpp` | CAQEncoder | Scalar quantize + code adjustment + factor computation |
| `fast_scan.cpp` | — | AVX-512/AVX2 fastscan primitives (pack, accumulate, LUT) |
| `lut.cpp` | Lut | SIMD lookup table for high-accuracy fast IP |
| `cluster_data.cpp` | CAQClusterData, SaqCluData | Per-segment and multi-segment cluster storage |
| `cluster_packer.cpp` | ClusterPacker | Pack short/long codes + factors into cluster format |
| `single_data.cpp` | CaqSingleDataWrapper | Single-vector data wrappers |
| `quantizer.cpp` | QuantizerCluster, QuantizerSingle | Per-segment batch and single-vector encoding |
| `saq_quantizer.cpp` | SAQuantizer | Multi-segment cluster encoding orchestrator |
| `caq_estimator.cpp` | CaqCluEstimator | 3-stage SIMD distance (variance, fastscan, accurate) |
| `saq_estimator.cpp` | SaqCluEstimator | Multi-segment distance aggregation |
| `saq_searcher.cpp` | SAQSearcher | Block-level 3-stage search with early termination |
| `ivf_index.cpp` | IVF | IVF index: construct, search, save/load |

## OpenMP Parallelization

Index construction (`IVF::construct`) parallelizes per-cluster encoding with `#pragma omp parallel for` (guarded by `SAQ_USE_OPENMP`).
