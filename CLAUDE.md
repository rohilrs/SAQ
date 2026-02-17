# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAQ (Scalar Additive Quantization) is a C++20 implementation of the algorithm from arXiv:2509.12086 for approximate nearest neighbor search. It uses CAQ (Code Adjustment Quantization) with per-vector v_max scaling, joint dynamic programming for dimension segmentation + bit allocation, IVF indexing, and SIMD-accelerated 3-stage search (variance pruning → 1-bit fastscan → accurate). The paper evaluates at 0.2-9 bits/dim (bpd) with PCA enabled and nprobe=200.

## Workflow

**Preprocessing (Python, using Faiss):**
```bash
python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k
python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k -K 4096
python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k
```

**Build C++ library + samples:**
```bash
cmake -B build -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build build
```

**Run benchmark:**
```bash
./build/samples/saq_dbpedia_sample data/datasets/dbpedia_100k results/saq 2.0 4096 200 8
# Args: data_dir results_dir bpd num_clusters nprobe num_threads
```

## Build Commands

### C++ Library + Samples (default)
```bash
cmake -B build -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build build
```

### With Tests
```bash
cmake -B build -DSAQ_BUILD_TESTS=ON
cmake --build build
```

### Python Bindings
```bash
cmake -B build -DSAQ_BUILD_PYTHON=ON
cmake --build build --target _saq_core
```

## CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `SAQ_BUILD_SAMPLES` | ON | Build sample programs |
| `SAQ_BUILD_TESTS` | OFF | Build unit tests |
| `SAQ_REQUIRE_AVX512` | ON | Enable AVX-512 SIMD (AVX2 fallback exists) |
| `SAQ_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `SAQ_BUILD_PYTHON` | OFF | Build pybind11 Python bindings |

## Architecture

### Algorithm Pipeline

**Preprocessing (Python):** PCA projection (Faiss PCAMatrix) → K-means clustering (Faiss IVF) → ground truth computation → output fvecs/ivecs files

**Index Construction (C++):** Load pre-computed data → SaqDataMaker (variance → joint DP → segment plan → per-segment rotators) → SAQuantizer (per-cluster: rotate → CAQ encode → pack short/long codes) → IVF index

**Search (C++):** FlatInitializer (nprobe nearest centroids) → SAQSearcher per cluster:
1. Variance pruning (cheap lower bounds, skip blocks > distk)
2. 1-bit fastscan (SIMD LUT lookup for 32 vectors at once)
3. Accurate distance (full-bit codes for promising candidates)

### Source Layout

Headers in `include/saq/` + `include/index/`, implementations in `src/`:

**Core Quantization:**
- **`quantization_plan`** — SaqData (plan container), SaqDataMaker (DP optimizer), BaseQuantizerData (per-segment metadata)
- **`caq_encoder`** — CAQEncoder (scalar quantize + code adjustment), CaqCode, QuantBaseCode
- **`quantizer`** — QuantizerCluster (batch encode), QuantizerSingle (single-vector encode)
- **`saq_quantizer`** — SAQuantizer (multi-segment cluster orchestrator), SAQuantizerSingle

**Search:**
- **`caq_estimator`** — CaqCluEstimator (3-stage SIMD distance), CaqEstimatorSingleImpl (non-fastscan)
- **`saq_estimator`** — SaqCluEstimator (multi-segment aggregator), SaqEstimatorBase
- **`saq_searcher`** — SAQSearcher (block-level 3-stage search with early termination)

**Data Structures:**
- **`cluster_data`** — CAQClusterData (per-segment storage), SaqCluData (multi-segment wrapper)
- **`cluster_packer`** — ClusterPacker (encode vectors → packed cluster format)
- **`single_data`** — CaqSingleDataWrapper, SaqSingleDataWrapper (single-vector storage)
- **`lut`** — Lut (SIMD lookup table for fast IP estimation)

**Index:**
- **`ivf_index`** — IVF (inverted file index with construct/search/save/load)
- **`initializer`** — FlatInitializer (brute-force centroid search)

**Utilities:**
- **`defines`** — Eigen types, constants, enums (DistType, BaseQuantType, Candidate)
- **`config`** — QuantizeConfig, QuantSingleConfig, SearcherConfig
- **`tools`** — Math utilities, SIMD L2/IP, binary operations
- **`code_helper`** — Bit-packed code operations (1-16 bits), SIMD dot products
- **`fast_scan`** — AVX-512/AVX2 fastscan primitives (pack, accumulate, LUT)
- **`memory`** — Aligned allocation, prefetch
- **`pool`** — ResultPool (top-K), AvgMaxRecorder (statistics)
- **`rotator`** — Rotator (random orthogonal), PCARotator
- **`io_utils`** — fvecs/ivecs/bin file I/O with Eigen matrices
- **`stopw`** — Stopwatch timer

### Key Data Structures

```
SaqData: QuantizeConfig + quant_plan[(dim_len, bits)] + BaseQuantizerData[] + data_variance
SaqCluData: CAQClusterData[] (segments) with short_codes + long_codes + factors + IDs
IVF: Initializer + SaqData + SaqCluData[] (per-cluster) + QuantizeConfig
```

### Python Preprocessing

Scripts in `python/preprocessing/`:
- **`pca.py`** — PCA transform via Faiss PCAMatrix, outputs `vectors_pca.fvecs`, `queries_pca.fvecs`, `variances_pca.fvecs`
- **`ivf.py`** — K-means clustering via Faiss, outputs `centroids_{K}.fvecs`, `cluster_ids_{K}.ivecs`
- **`compute_gt.py`** — Brute-force ground truth, outputs `groundtruth.ivecs`

## Adding New Components

1. Add header to `include/saq/` (or `include/index/`)
2. Add implementation to `src/`
3. Add source file to `src/CMakeLists.txt` via `target_sources(saq PRIVATE ...)`
4. Add test in `tests/` and register in `tests/CMakeLists.txt`

## Dependencies

All C++ dependencies are fetched automatically via CMake FetchContent:

- **Eigen3** 3.4.0 (FetchContent)
- **glog** 0.7.1 (FetchContent)
- **gflags** 2.2.2 (FetchContent)
- **fmt** 10.2.1 (FetchContent)
- **OpenMP** (system, optional)
- **pybind11** v2.12.0 (FetchContent, only when `SAQ_BUILD_PYTHON=ON`)
- **Python**: numpy>=1.20.0, faiss-cpu>=1.7.0 (for preprocessing)

## Conventions

- C++20 required; MSVC uses `/arch:AVX512`, GCC/Clang uses `-mavx512f -mavx512bw -mavx512dq -mavx512vl`
- Namespace: `saq` (reference repo uses `saqlib`)
- Most classes are header-only (inline in .h); .cpp files are minimal translation units
- OpenMP parallelization guarded by `SAQ_USE_OPENMP` compile definition
- Tests are standalone executables linked against the `saq` static library
