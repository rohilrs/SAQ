# SAQ: Scalar Additive Quantization

A high-performance C++17 implementation of **Scalar Additive Quantization** for approximate nearest neighbor search, based on the paper [arXiv:2509.12086](https://arxiv.org/abs/2509.12086).

## Features

- **Extreme Compression**: 768x compression ratio (1536d float32 → 64 bits)
- **Scalar Quantization**: Uniform grid per dimension with per-vector v_max scaling
- **Joint DP Optimization**: Single dynamic programming pass for simultaneous segmentation and bit allocation
- **IVF Indexing**: Scalable to 1M-10M+ vectors with sublinear search
- **OpenMP Parallelization**: Multi-threaded build and search (5.6x speedup at 8 threads)
- **Python Bindings**: pybind11 interface with NumPy interop (~3% overhead vs C++)
- **SIMD Acceleration**: AVX-512/AVX2 kernels for distance computation
- **Code Adjustment (CAQ)**: Per-dimension refinement optimizing cosine similarity

## Quick Start

### Build

```powershell
# Windows (MSVC with AVX-512)
mkdir build && cd build
cmake .. -G Ninja -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build .
```

```bash
# Linux/macOS
mkdir -p build && cd build
cmake .. -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
make -j$(nproc)
```

### Build with Python Bindings

```bash
cmake .. -DSAQ_BUILD_PYTHON=ON
cmake --build . --target _saq_core
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
│   │   ├── bit_allocation_dp.h  # Joint segmentation + bit allocation
│   │   ├── caq_code_adjustment.h # Per-dimension code adjustment
│   │   ├── distance_estimator.h # Scalar IP estimation
│   │   └── simd_kernels.h       # AVX-512/AVX2 kernels
│   └── index/         # Indexing structures
│       ├── ivf_index.h          # IVF partitioning
│       └── fast_scan/           # SIMD-accelerated scanning
├── src/               # Implementation files
├── samples/           # End-to-end examples
├── python/
│   ├── saq/           # Python package (pybind11 bindings)
│   ├── bindings/      # C++/Python binding source
│   ├── ivf_clustering.py  # FAISS-based clustering utility
│   └── benchmark_saq.py   # Python benchmark script
├── data/              # Datasets (downloaded, not in git)
├── results/           # Benchmark results
├── tests/             # Unit tests
└── docs/              # Documentation and diagrams
```

## Algorithm Overview

```
Training:
  1. PCA Projection (optional)        → reduce dimensionality
  2. Joint DP (segmentation + bits)   → optimal segments and bit allocation
  3. Per-segment random rotation      → decorrelate dimensions within segments
  4. Compute per-dimension statistics → variance for distortion model

Encoding:
  1. Project vector (PCA if enabled)
  2. Apply per-segment rotation
  3. Scalar quantize: c[i] = floor((v[i] + v_max) / delta)
  4. CAQ refinement (optional) → per-dimension ±1 adjustments for cosine similarity

Search:
  1. Find nprobe nearest clusters (IVF)
  2. Precompute scalar query table
  3. Estimate IP: <o,q> = delta * <codes, q> + q_sum * (-v_max + delta/2)
  4. Convert to L2: ||q-o||^2 = ||q||^2 - 2*IP + ||o||^2
  5. Return top-K results
```

See [docs/saq_codeflow.md](docs/saq_codeflow.md) for detailed Mermaid diagrams.

## Requirements

- **Compiler**: C++17 (MSVC 2019+, GCC 9+, Clang 10+)
- **SIMD**: AVX-512 recommended, AVX2 fallback supported
- **CMake**: 3.16+
- **OpenMP** (optional): For multi-threaded build and search
- **Python** (optional): 3.8+ with numpy for bindings; faiss for clustering

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SAQ_BUILD_SAMPLES` | ON | Build sample programs |
| `SAQ_BUILD_TESTS` | OFF | Build unit tests |
| `SAQ_REQUIRE_AVX512` | ON | Require AVX-512 support |
| `SAQ_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `SAQ_BUILD_PYTHON` | OFF | Build Python bindings (pybind11) |

## Citation

```bibtex
@article{saq2025,
  title     = {SAQ: Scalar Additive Quantization},
  eprint    = {2509.12086},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.12086}
}
```
