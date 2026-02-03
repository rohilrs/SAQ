# SAQ Include Directory

Public C++ headers for the SAQ library.

## Directory Structure

```
include/
├── saq/           # Core SAQ algorithm
│   ├── saq_quantizer.h          # Main interface
│   ├── quantization_plan.h      # Serializable plan
│   ├── pca_projection.h         # Dimensionality reduction
│   ├── dimension_segmentation.h # Variance-based partitioning
│   ├── bit_allocation_dp.h      # Optimal bit distribution
│   ├── caq_code_adjustment.h    # Cross-segment refinement
│   ├── distance_estimator.h     # Asymmetric distance tables
│   └── simd_kernels.h           # AVX-512/AVX2 primitives
│
├── index/         # Indexing structures
│   ├── ivf_index.h              # IVF partitioning for scale
│   └── fast_scan/
│       └── fast_scan.h          # SIMD-accelerated scanning
│
└── third/         # Third-party dependencies (header-only)
```

## Component Overview

### saq/ — Core Quantization

| Header | Purpose |
|--------|---------|
| `saq_quantizer.h` | Master orchestrator: Train, Encode, Decode, Search |
| `quantization_plan.h` | Data structures for serialization (PCA, segments, codebooks) |
| `pca_projection.h` | Optional SVD-based dimensionality reduction |
| `dimension_segmentation.h` | Partitions dimensions into contiguous segments |
| `bit_allocation_dp.h` | Dynamic programming for rate-distortion optimal bit allocation |
| `caq_code_adjustment.h` | Greedy code refinement to minimize reconstruction error |
| `distance_estimator.h` | Precomputes query-to-centroid distance tables |
| `simd_kernels.h` | L2 distance, inner product with AVX-512/AVX2 |

### index/ — Scalable Search

| Header | Purpose |
|--------|---------|
| `ivf_index.h` | Inverted File Index for sublinear search (1M+ vectors) |
| `fast_scan/fast_scan.h` | FAISS-style FastScan with vpshufb for 1.6-2x speedup |

## Usage Example

```cpp
#include "saq/saq_quantizer.h"
#include "index/ivf_index.h"

using namespace saq;

// Train quantizer
SAQQuantizer quantizer;
SAQTrainConfig config;
config.total_bits = 64;
config.num_segments = 16;
quantizer.Train(data, n_vectors, dim, config);

// Build IVF index
IVFIndex index;
IVFTrainConfig ivf_config;
ivf_config.ivf.num_clusters = 1024;
ivf_config.ivf.use_fast_scan = true;
index.Build(data, n_vectors, dim, centroids, assignments, ivf_config);

// Search
std::vector<std::vector<IVFSearchResult>> results;
index.SearchBatch(queries, n_queries, k, results, nprobe);
```

## Include Order

Headers are designed to be self-contained. Include what you need:

```cpp
// Minimal: just quantizer
#include "saq/saq_quantizer.h"

// Full index with FastScan
#include "index/ivf_index.h"  // Includes saq_quantizer.h transitively
```
