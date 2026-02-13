# SAQ Python

Python bindings, utilities, and benchmarks for the SAQ library.

## Setup

```bash
# Install numpy (required for bindings)
pip install numpy

# For clustering support
pip install faiss-cpu
```

## Python Bindings (pybind11)

The `saq` package provides Python access to the C++ SAQ library with NumPy interop.

### Building

```bash
# From project root
mkdir build && cd build
cmake .. -DSAQ_BUILD_PYTHON=ON
cmake --build . --target _saq_core
```

The compiled module is placed in `python/saq/` automatically.

### Usage

```python
import sys
sys.path.insert(0, "python")  # Or install with pip install -e python/

import numpy as np
import saq

# Create and build IVF index
index = saq.IVFIndex()
config = saq.IVFTrainConfig()
config.ivf.num_clusters = 100
config.saq.total_bits = 64

data = np.random.randn(10000, 128).astype(np.float32)
centroids = ...  # From k-means
assignments = ...  # Cluster IDs

index.build(data, centroids, assignments, config)

# Search
queries = np.random.randn(10, 128).astype(np.float32)
indices, distances = index.search_batch(queries, k=10, nprobe=10)

# Save/Load
index.save("my_index.bin")
index2 = saq.IVFIndex()
index2.load("my_index.bin")
```

### Available Classes

| Class | Description |
|-------|-------------|
| `saq.SAQQuantizer` | Train, encode_batch, decode, search |
| `saq.IVFIndex` | build, search, search_batch, save, load, reconstruct |
| `saq.SAQTrainConfig` | Training parameters (total_bits, use_pca, metric, etc.) |
| `saq.SAQEncodeConfig` | Encoding parameters (use_caq) |
| `saq.IVFConfig` | IVF parameters (num_clusters, nprobe, hnsw) |
| `saq.IVFTrainConfig` | Combined IVF + SAQ config |
| `saq.DistanceMetric` | L2 or InnerProduct |

### Performance

The GIL is released during compute-heavy C++ calls (`build`, `search_batch`, `encode_batch`), so Python binding overhead is minimal (~3% on search QPS).

## Scripts

### ivf_clustering.py — K-means Clustering for IVF

Performs K-means clustering using FAISS to produce centroids and assignments
needed for IVF index construction.

```bash
python ivf_clustering.py <data_file> <num_clusters> <centroids_out> <cluster_ids_out> [metric]

# Example: Cluster SIFT1M into 4096 partitions
python ivf_clustering.py \
    data/datasets/sift/sift_base.fvecs \
    4096 \
    data/datasets/sift/centroids.fvecs \
    data/datasets/sift/cluster_ids.ivecs \
    l2
```

### benchmark_saq.py — Python Benchmark

Benchmarks IVF search performance through the Python bindings.

```bash
# Set PYTHONPATH to include python/ directory
export PYTHONPATH=python:$PYTHONPATH

# Run all configurations
python python/benchmark_saq.py

# Small-scale only
python python/benchmark_saq.py --small-only
```

### samples/download_dbpedia.py — Dataset Download

Downloads the DBpedia 100K dataset from HuggingFace.

```bash
pip install datasets
python samples/download_dbpedia.py
```

## Requirements

See [requirements.txt](requirements.txt):

```
numpy>=1.20
faiss-cpu>=1.7.0    # Or faiss-gpu for CUDA support
```

## Integration with C++

```
┌─────────────────────┐     ┌─────────────────────┐
│  download_dbpedia.py│────>│  vectors.fvecs      │
│                     │     │  queries.fvecs      │
│                     │     │  groundtruth.ivecs  │
└─────────────────────┘     └──────────┬──────────┘
                                       │
┌─────────────────────┐                v
│  ivf_clustering.py  │────>  centroids.fvecs
│  (optional)         │       cluster_ids.ivecs
└─────────────────────┘                │
                                       v
                  ┌──────────────────────────────────┐
                  │  saq_dbpedia_sample (C++)         │
                  │  — or —                          │
                  │  saq.IVFIndex (Python bindings)   │
                  └──────────────────────────────────┘
```
