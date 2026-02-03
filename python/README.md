# SAQ Python Utilities

Python scripts for data preparation, clustering, and benchmarking support.

## Setup

```bash
# Create conda environment (recommended)
conda create -n saq python=3.10
conda activate saq

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy faiss-cpu datasets
```

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

**Arguments:**
| Argument | Description |
|----------|-------------|
| `data_file` | Input vectors (.fvecs or .npy) |
| `num_clusters` | Number of clusters K (recommend: 4 × √n) |
| `centroids_out` | Output centroid vectors |
| `cluster_ids_out` | Output cluster assignments |
| `metric` | `l2` (default) or `ip` (inner product) |

**Why use this instead of C++ clustering?**
- FAISS provides GPU-accelerated k-means
- Faster for large datasets (1M+ vectors)
- Pre-clustering separates concerns from index building

### samples/download_dbpedia.py — Dataset Download

Downloads the DBpedia 100K dataset from HuggingFace and prepares it for benchmarking.

```bash
python samples/download_dbpedia.py
```

**Output:**
- `data/datasets/dbpedia_100k/vectors.fvecs` — 99K base vectors (1536d)
- `data/datasets/dbpedia_100k/queries.fvecs` — 1K query vectors
- `data/datasets/dbpedia_100k/groundtruth.ivecs` — Exact k-NN (k=100)

## File Format Utilities

Both scripts include utility functions for reading/writing standard formats:

```python
from ivf_clustering import read_fvecs, write_fvecs, read_ivecs, write_ivecs

# Read vectors
vectors = read_fvecs("data/sift_base.fvecs")  # Returns np.ndarray (n, d)

# Write vectors
write_fvecs("output.fvecs", vectors)

# Read ground truth
gt = read_ivecs("groundtruth.ivecs")  # Returns np.ndarray (n, k)
```

## Requirements

See [requirements.txt](requirements.txt):

```
numpy>=1.20
faiss-cpu>=1.7.0    # Or faiss-gpu for CUDA support
datasets>=2.0       # For HuggingFace dataset loading
```

## Integration with C++

The Python scripts produce files that the C++ sample directly consumes:

```
┌─────────────────────┐     ┌─────────────────────┐
│  download_dbpedia.py│────▶│  vectors.fvecs      │
│                     │     │  queries.fvecs      │
│                     │     │  groundtruth.ivecs  │
└─────────────────────┘     └──────────┬──────────┘
                                       │
┌─────────────────────┐                ▼
│  ivf_clustering.py  │────▶  centroids.fvecs
│  (optional)         │       cluster_ids.ivecs
└─────────────────────┘                │
                                       ▼
                            ┌─────────────────────┐
                            │  saq_dbpedia_sample │
                            │  (C++ executable)   │
                            └─────────────────────┘
```
