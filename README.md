# SAQ: Scalar Additive Quantization

A C++20 implementation of **Scalar Additive Quantization** for approximate nearest neighbor search, based on arXiv:2509.12086 and aligned with the [reference repository](https://github.com/howarlii/SAQ).

## Features

- **Extreme Compression**: 8x-32x compression (e.g., 1536d float32 at 1-4 bits/dim)
- **CAQ Encoding**: Per-vector, per-segment v_max scaling with code adjustment (cosine optimization)
- **Joint DP Optimization**: Dynamic programming over 64-dim blocks for segment boundaries + bit allocation
- **3-Stage Search**: Variance pruning -> 1-bit fastscan (SIMD) -> accurate distance with early termination
- **IVF Indexing**: Inverted file index for sublinear search
- **SIMD Acceleration**: AVX-512/AVX2 LUT-based fastscan (32 vectors at once)
- **OpenMP Parallelization**: Multi-threaded index construction

## Quick Start

### Prerequisites (Python)

Preprocessing requires Python with Faiss:

```bash
pip install numpy faiss-cpu
```

### Preprocess Dataset

```bash
cd python
python -m preprocessing.pca --data-dir ../data/datasets/dbpedia_100k
python -m preprocessing.ivf --data-dir ../data/datasets/dbpedia_100k -K 4096
python -m preprocessing.compute_gt --data-dir ../data/datasets/dbpedia_100k
```

### Build

```bash
cmake -B build -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build build
```

### Run Benchmark

```bash
./build/samples/saq_dbpedia_sample data/datasets/dbpedia_100k results/saq 2.0 4096 200 8
# Args: data_dir results_dir bpd num_clusters nprobe num_threads
```

## Benchmark Results

**Dataset:** DBpedia 100K (1536D OpenAI embeddings, K=4096, nprobe=200)

| bpd | Compression | R@1 | R@10 | R@100 |
|-----|-------------|-----|------|-------|
| 1.0 | 32x | 85.0% | 87.3% | 86.6% |
| 2.0 | 16x | 92.8% | 92.6% | 90.0% |
| 4.0 | 8x | 97.0% | 94.8% | 90.9% |

## Directory Structure

```
SAQ/
├── include/
│   ├── saq/                    # Core SAQ algorithm headers
│   │   ├── quantization_plan.h     # SaqData, SaqDataMaker (DP optimizer)
│   │   ├── caq_encoder.h           # CAQEncoder (quantize + code adjustment)
│   │   ├── quantizer.h             # QuantizerCluster, QuantizerSingle
│   │   ├── saq_quantizer.h         # SAQuantizer (multi-segment orchestrator)
│   │   ├── caq_estimator.h         # CaqCluEstimator (3-stage distance)
│   │   ├── saq_estimator.h         # SaqCluEstimator (multi-segment aggregator)
│   │   ├── saq_searcher.h          # SAQSearcher (block-level 3-stage search)
│   │   ├── cluster_data.h          # CAQClusterData, SaqCluData
│   │   ├── cluster_packer.h        # ClusterPacker (code + factor packing)
│   │   ├── single_data.h           # Single-vector data wrappers
│   │   ├── fast_scan.h             # AVX-512/AVX2 fastscan primitives
│   │   ├── lut.h                   # SIMD lookup table for fast IP
│   │   ├── rotator.h               # Per-segment random orthogonal rotation
│   │   └── ...                     # config, defines, tools, memory, etc.
│   └── index/
│       └── ivf_index.h             # IVF (construct, search, save/load)
├── src/                        # Implementation files (.cpp)
├── samples/                    # Benchmark program
├── python/
│   ├── preprocessing/          # PCA, K-means, ground truth (Faiss)
│   └── bindings/               # pybind11 bindings (needs updating)
├── data/                       # Datasets (not in git)
└── docs/                       # Architecture diagrams
```

## Algorithm Overview

### Preprocessing (Python, Faiss)

1. PCA full rotation -> reorder dimensions by variance
2. K-means clustering -> IVF partitions
3. Brute-force ground truth computation

### Index Construction (C++)

1. Load PCA-transformed data, centroids, cluster assignments
2. `SaqDataMaker`: compute variance -> joint DP over 64-dim blocks -> segment plan + per-segment rotators
3. `SAQuantizer` per cluster: rotate -> `CAQEncoder` (quantize + code adjustment) -> `ClusterPacker` (pack short/long codes)
4. Build `IVF` index

### Search (C++)

1. `FlatInitializer`: find nprobe nearest centroids
2. `SAQSearcher` per cluster (3-stage, block-level):
   - **Variance pruning**: cheap lower bounds, skip blocks exceeding distk
   - **1-bit fastscan**: SIMD LUT lookup on MSB codes (32 vectors at once)
   - **Accurate distance**: full-bit codes for promising candidates

## Requirements

- **Compiler**: C++20 (MSVC 2019+, GCC 10+, Clang 12+)
- **SIMD**: AVX-512 required (AVX2 fallback for some operations)
- **CMake**: 3.16+
- **OpenMP** (optional): For multi-threaded index construction
- **Python** (preprocessing): 3.8+ with numpy, faiss-cpu

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SAQ_BUILD_SAMPLES` | ON | Build sample programs |
| `SAQ_BUILD_TESTS` | OFF | Build unit tests |
| `SAQ_REQUIRE_AVX512` | ON | Enable AVX-512 SIMD |
| `SAQ_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `SAQ_BUILD_PYTHON` | OFF | Build Python bindings (pybind11) |

## Dependencies

All C++ dependencies are fetched automatically via CMake FetchContent:

- **Eigen3** 3.4.0
- **glog** 0.7.1
- **gflags** 2.2.2
- **fmt** 10.2.1
- **pybind11** 2.12.0 (only when `SAQ_BUILD_PYTHON=ON`)

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
