# SAQ: Segmented Additive Quantization

A high-performance C++17 implementation of **Segmented Additive Quantization** for approximate nearest neighbor search, based on the paper [arXiv:2509.12086](https://arxiv.org/abs/2509.12086).

## Features

- **Extreme Compression**: 768x compression ratio (1536d float32 → 64 bits)
- **IVF Indexing**: Scalable to 1M-10M+ vectors with sublinear search
- **FastScan SIMD**: AVX-512/AVX2 accelerated distance estimation (1.6-2.1x speedup)
- **Asymmetric Distance**: Precomputed lookup tables for fast query processing
- **Flexible Bit Allocation**: Dynamic programming for optimal bits-per-segment

## Quick Start

### Build

```powershell
# Windows (MSVC with AVX-512)
mkdir build && cd build
cmake .. -G Ninja -DSAQ_BUILD_SAMPLES=ON
cmake --build .
```

```bash
# Linux/macOS
mkdir -p build && cd build
cmake .. -DSAQ_BUILD_SAMPLES=ON
make -j$(nproc)
```

### Run Sample

```bash
# Download dataset (requires Python with datasets, numpy)
python samples/download_dbpedia.py

# Run benchmark
./build/samples/saq_dbpedia_sample
```

## Directory Structure

```
SAQ/
├── include/
│   ├── saq/           # Core SAQ algorithm headers
│   │   ├── saq_quantizer.h      # Main quantizer interface
│   │   ├── quantization_plan.h  # Serializable plan structure
│   │   ├── pca_projection.h     # Dimensionality reduction
│   │   ├── dimension_segmentation.h
│   │   ├── bit_allocation_dp.h  # Optimal bit distribution
│   │   ├── caq_code_adjustment.h
│   │   ├── distance_estimator.h # Asymmetric distance
│   │   └── simd_kernels.h       # AVX-512/AVX2 kernels
│   └── index/         # Indexing structures
│       ├── ivf_index.h          # IVF partitioning
│       └── fast_scan/           # SIMD-accelerated scanning
├── src/               # Implementation files
├── samples/           # End-to-end examples
├── python/            # Python utilities (clustering, data prep)
├── data/              # Datasets (downloaded, not in git)
├── results/           # Benchmark results
├── tests/             # Unit tests
└── docs/              # Documentation and diagrams
```

## Performance (DBpedia 100K, 1536d)

| nprobe | Recall@100 | QPS    | Compression |
|--------|------------|--------|-------------|
| 1      | 27.7%      | 1,284  | 768x        |
| 8      | 26.9%      | 1,166  | 768x        |
| 32     | 23.8%      | 1,017  | 768x        |
| 128    | 22.6%      | 748    | 768x        |

*64 bits per vector, FastScan enabled, single-threaded*

## Algorithm Overview

```
Training:
  1. PCA Projection (optional) → reduce dimensionality
  2. Dimension Segmentation   → partition dims by variance
  3. Bit Allocation (DP)      → optimal bits per segment
  4. Codebook Training        → k-means per segment

Encoding:
  1. Project vector (if PCA)
  2. Quantize each segment → nearest centroid
  3. CAQ Refinement (optional) → cross-segment optimization

Search:
  1. Find nprobe nearest clusters (IVF)
  2. Precompute distance tables (query → centroids)
  3. Scan clusters with table lookups (FastScan SIMD)
  4. Return top-K results
```

See [docs/saq_codeflow.md](docs/saq_codeflow.md) for detailed Mermaid diagrams.

## Requirements

- **Compiler**: C++17 (MSVC 2019+, GCC 9+, Clang 10+)
- **SIMD**: AVX-512 recommended, AVX2 fallback supported
- **CMake**: 3.16+
- **Python** (optional): 3.8+ with numpy, datasets, faiss

## Citation

```bibtex
@article{saq2025,
  title     = {SAQ: Segmented Additive Quantization},
  eprint    = {2509.12086},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.12086}
}
```
