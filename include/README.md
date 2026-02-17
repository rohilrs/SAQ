# SAQ Include Directory

Public C++ headers for the SAQ library.

## Directory Structure

```
include/
├── saq/                        # Core SAQ algorithm
│   ├── quantization_plan.h         # SaqData (plan), SaqDataMaker (DP optimizer)
│   ├── caq_encoder.h               # CAQEncoder (quantize + code adjustment)
│   ├── quantizer.h                 # QuantizerCluster, QuantizerSingle
│   ├── saq_quantizer.h             # SAQuantizer (multi-segment orchestrator)
│   ├── caq_estimator.h             # CaqCluEstimator (3-stage SIMD distance)
│   ├── saq_estimator.h             # SaqCluEstimator (multi-segment aggregator)
│   ├── saq_searcher.h              # SAQSearcher (block-level 3-stage search)
│   ├── cluster_data.h              # CAQClusterData, SaqCluData
│   ├── cluster_packer.h            # ClusterPacker (code + factor packing)
│   ├── single_data.h               # CaqSingleDataWrapper, SaqSingleDataWrapper
│   ├── fast_scan.h                 # AVX-512/AVX2 fastscan primitives
│   ├── lut.h                       # Lut (SIMD lookup table for fast IP)
│   ├── rotator.h                   # Rotator (random orthogonal), PCARotator
│   ├── config.h                    # QuantizeConfig, SearcherConfig
│   ├── defines.h                   # Eigen types, constants, enums (DistType)
│   ├── tools.h                     # Math utilities, SIMD L2/IP
│   ├── code_helper.h               # Bit-packed code operations (1-16 bits)
│   ├── memory.h                    # Aligned allocation, portable_aligned_free
│   ├── pool.h                      # ResultPool (top-K), AvgMaxRecorder
│   ├── io_utils.h                  # fvecs/ivecs/bin file I/O
│   └── stopw.h                     # Stopwatch timer
│
└── index/
    └── ivf_index.h                 # IVF (construct, search, save/load)
```

## Component Overview

### Core Quantization

| Header | Purpose |
|--------|---------|
| `quantization_plan.h` | SaqDataMaker: joint DP over 64-dim blocks for segment boundaries + bit allocation. SaqData: serializable plan container. |
| `caq_encoder.h` | CAQEncoder: scalar quantize, code adjustment (cosine optimization), factor computation (fac_rescale, fac_err). |
| `quantizer.h` | QuantizerCluster (batch encode per segment), QuantizerSingle (single-vector encode). |
| `saq_quantizer.h` | SAQuantizer: multi-segment orchestrator for per-cluster encoding. |

### Search

| Header | Purpose |
|--------|---------|
| `caq_estimator.h` | CaqCluEstimator: 3-stage SIMD distance (variance pruning, 1-bit fastscan, accurate). |
| `saq_estimator.h` | SaqCluEstimator: aggregates per-segment estimators for multi-segment distance. |
| `saq_searcher.h` | SAQSearcher: block-level (32 vectors) 3-stage search with early termination. |
| `fast_scan.h` | AVX-512/AVX2 fastscan: pack codes, LUT lookup, accumulate distances. |
| `lut.h` | Lut: high-accuracy fast IP computation via precomputed lookup tables. |

### Data Structures

| Header | Purpose |
|--------|---------|
| `cluster_data.h` | CAQClusterData (per-segment: short_codes, long_codes, factors), SaqCluData (multi-segment wrapper). |
| `cluster_packer.h` | ClusterPacker: encode vectors into packed cluster format (fastscan-friendly). |
| `single_data.h` | CaqSingleDataWrapper, SaqSingleDataWrapper for single-vector storage. |

### Index

| Header | Purpose |
|--------|---------|
| `ivf_index.h` | IVF class: construct index from preprocessed data, 3-stage search, binary save/load. |
| `initializer.h` | FlatInitializer: brute-force centroid search for nprobe selection. |
