# SAQ Include Directory

Public C++ headers for the SAQ library.

## Directory Structure

```
include/
├── saq/           # Core SAQ algorithm
│   ├── saq_quantizer.h          # Main interface
│   ├── quantization_plan.h      # Serializable plan
│   ├── pca_projection.h         # Dimensionality reduction
│   ├── dimension_segmentation.h # Per-dimension variance stats
│   ├── bit_allocation_dp.h      # Joint segmentation + bit allocation DP
│   ├── caq_code_adjustment.h    # Per-dimension code adjustment (cosine)
│   ├── distance_estimator.h     # Scalar IP estimation (paper formula)
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
| `saq_quantizer.h` | Master orchestrator: Train, Encode, Decode, Search using scalar quantization |
| `quantization_plan.h` | Serializable container (PCA, segments, codebooks, rotation matrices). Version 2 supports scalar quantization |
| `pca_projection.h` | Optional SVD-based dimensionality reduction |
| `dimension_segmentation.h` | Computes per-dimension statistics (mean, variance) |
| `bit_allocation_dp.h` | Joint DP: simultaneously determines segment boundaries and bits-per-dimension allocation |
| `caq_code_adjustment.h` | Per-dimension code adjustment (Algorithm 1 from paper): tries +/-1 changes to optimize cosine similarity |
| `distance_estimator.h` | Scalar IP estimation: `<o,q> = delta * <codes, q> + q_sum * (-v_max + delta/2)`. Also supports legacy codebook-based ADC |
| `simd_kernels.h` | L2 distance, inner product, batch operations with AVX-512/AVX2 |

### index/ — Scalable Search

| Header | Purpose |
|--------|---------|
| `ivf_index.h` | Inverted File Index for sublinear search (1M+ vectors). Stores per-dimension scalar codes + v_max per vector. L2 distance from IP estimation |
| `fast_scan/fast_scan.h` | FAISS-style FastScan with vpshufb for packed code lookup |

## Usage Example

```cpp
#include "saq/saq_quantizer.h"
#include "index/ivf_index.h"

using namespace saq;

// Train quantizer
SAQQuantizer quantizer;
SAQTrainConfig config;
config.total_bits = 64;
config.use_segment_rotation = true;
quantizer.Train(data, n_vectors, dim, config);

// Encode vectors
std::vector<ScalarEncodedVector> encoded;
quantizer.EncodeBatch(data, n_vectors, encoded);

// Build IVF index
IVFIndex index;
IVFTrainConfig ivf_config;
ivf_config.ivf.num_clusters = 1024;
ivf_config.saq.total_bits = 64;
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

// Full index
#include "index/ivf_index.h"  // Includes saq_quantizer.h transitively
```
