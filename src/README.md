# SAQ Source Files

Implementation files for the SAQ library.

## Files

| File | Description |
|------|-------------|
| `saq_quantizer.cpp` | Main quantizer: Train, Encode, Decode, Search with scalar quantization |
| `bit_allocation_dp.cpp` | Joint DP for simultaneous segmentation and bit allocation |
| `dimension_segmentation.cpp` | Per-dimension variance statistics (Welford's algorithm) |
| `pca_projection.cpp` | SVD-based dimensionality reduction (OpenMP-parallelized) |
| `caq_code_adjustment.cpp` | Per-dimension code adjustment optimizing cosine similarity |
| `distance_estimator.cpp` | Scalar IP estimation using paper formula + legacy ADC |
| `quantization_plan.cpp` | Binary/JSON serialization (v1 codebook, v2 scalar+rotations) |
| `ivf_index.cpp` | IVF index: Build, Search, Save/Load with L2 from IP |
| `simd_kernels.cpp` | AVX-512/AVX2 distance computation kernels |
| `fast_scan.cpp` | SIMD-accelerated cluster scanning with packed codes |

## OpenMP Parallelization

The following functions use `#pragma omp parallel for` (guarded by `SAQ_USE_OPENMP`):

- `saq_quantizer.cpp`: `EncodeBatch()` — parallel per-vector encoding
- `distance_estimator.cpp`: `EstimateScalarIPBatch()`, `EstimateDistancesBatch()`
- `pca_projection.cpp`: `ComputeCovariance()`, `ProjectBatch()`
- `dimension_segmentation.cpp`: `ComputeStats()` — thread-local accumulators
- `ivf_index.cpp`: `Build()` encoding loop, `SearchBatch()` — parallel per-query
