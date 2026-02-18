# SAQ Python

Python preprocessing scripts and bindings for the SAQ library.

## Setup

```bash
pip install numpy faiss-cpu
```

## Preprocessing Scripts

The preprocessing pipeline prepares data for the C++ SAQ index. All scripts are in `preprocessing/`.

### 1. PCA Rotation

```bash
python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k
```

Applies full PCA rotation via `faiss.PCAMatrix`. Outputs:
- `vectors_pca.fvecs` — PCA-transformed base vectors
- `queries_pca.fvecs` — PCA-transformed query vectors
- `centroids_{K}_pca.fvecs` — PCA-transformed centroids (if centroids exist)
- `variances_pca.fvecs` — Per-dimension variance of transformed data

Options: `--pca-dim` (default: full rotation), `--base-file`, `--query-file`, `--centroids`

### 2. K-means Clustering

```bash
python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k -K 4096
```

Performs K-means via `faiss.IndexIVFFlat`. Outputs:
- `centroids_{K}.fvecs` — Cluster centroids
- `cluster_ids_{K}.ivecs` — Per-vector cluster assignments

Options: `-K` (num clusters, default 4096), `--metric` (l2 or ip)

### 3. Ground Truth

```bash
python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k
```

Brute-force exact kNN computation. Outputs:
- `groundtruth.ivecs` — Top-1000 nearest neighbor IDs per query

Options: `--top-k` (default 1000), `--threads` (default 8)

### Complete Pipeline

```bash
cd python
python -m preprocessing.pca --data-dir ../data/datasets/dbpedia_100k
python -m preprocessing.ivf --data-dir ../data/datasets/dbpedia_100k -K 4096
python -m preprocessing.pca --data-dir ../data/datasets/dbpedia_100k --centroids centroids_4096
python -m preprocessing.compute_gt --data-dir ../data/datasets/dbpedia_100k
```

The C++ sample expects: `vectors_pca.fvecs`, `queries_pca.fvecs`, `centroids_{K}_pca.fvecs`, `cluster_ids_{K}.ivecs`, `variances_pca.fvecs`, `groundtruth.ivecs`

## Python Bindings (pybind11)

The bindings in `bindings/saq_bindings.cpp` wrap the C++ SAQ library for use from Python via pybind11. numpy arrays are automatically converted to/from Eigen matrices.

### Building

```bash
# Point CMake to the conda/venv Python interpreter
cmake -B build -DSAQ_BUILD_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python)
cmake --build build --target _saq_core
```

This builds `python/saq/_saq_core.{version}.pyd` (Windows) or `.so` (Linux) and copies required DLLs (glog, fmt) next to it.

### Usage

```python
import sys; sys.path.insert(0, "python")
import saq
import numpy as np

# Configure quantization
cfg = saq.QuantizeConfig()
cfg.avg_bits = 4.0
cfg.enable_segmentation = True
cfg.single.quant_type = saq.BaseQuantType.CAQ
cfg.single.random_rotation = True
cfg.single.use_fastscan = True
cfg.single.caq_adj_rd_lmt = 6

# Build index
index = saq.IVF(n_vectors, dim, n_clusters, cfg)
index.set_variance(variances)  # optional: per-dim variance from PCA
index.construct(data, centroids, cluster_ids, num_threads=8)

# Search
scfg = saq.SearcherConfig()
scfg.dist_type = saq.DistType.L2Sqr
results = index.search_batch(queries, topk=10, nprobe=200, config=scfg)

# Save/load
index.save("index.bin")
index2 = saq.IVF()
index2.load("index.bin")
```

### Exposed API

| Python Class/Enum | C++ Type | Fields/Methods |
|-------------------|----------|----------------|
| `DistType` | `saq::DistType` | `L2Sqr`, `IP` |
| `BaseQuantType` | `saq::BaseQuantType` | `CAQ`, `RBQ`, `LVQ` |
| `QuantSingleConfig` | `saq::QuantSingleConfig` | `quant_type`, `random_rotation`, `use_fastscan`, `caq_adj_rd_lmt`, `caq_adj_eps` |
| `QuantizeConfig` | `saq::QuantizeConfig` | `avg_bits`, `enable_segmentation`, `use_compact_layout`, `single` |
| `SearcherConfig` | `saq::SearcherConfig` | `dist_type`, `searcher_vars_bound_m` |
| `IVF` | `saq::IVF` | `construct()`, `search()`, `search_batch()`, `save()`, `load()`, `set_variance()` |
| `load_fvecs()` | `saq::load_something<float>` | Load `.fvecs` file to numpy array |
| `load_ivecs()` | `saq::load_something<uint32_t>` | Load `.ivecs` file to numpy array |

## Utility Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `preprocessing/pca.py` | `preprocessing/` | PCA via Faiss PCAMatrix |
| `preprocessing/ivf.py` | `preprocessing/` | K-means via Faiss |
| `preprocessing/compute_gt.py` | `preprocessing/` | Brute-force ground truth |
| `preprocessing/utils/io.py` | `preprocessing/utils/` | Read/write fvecs/ivecs/fbin/ibin |
| `ivf_clustering.py` | root | Standalone K-means clustering utility |
| `benchmark_saq.py` | root | Python benchmark using bindings |

## Requirements

See [requirements.txt](requirements.txt):

```
numpy>=1.20.0
faiss-cpu>=1.7.0
```

## Integration with C++

```
Preprocessing (Python/Faiss)          C++ Index Construction + Search
┌──────────────────────┐              ┌──────────────────────────────┐
│  pca.py              │──────────>   │  saq_dbpedia_sample          │
│  ivf.py              │  .fvecs      │  (or custom C++ program)     │
│  compute_gt.py       │  .ivecs      │                              │
└──────────────────────┘              │  Loads preprocessed data     │
                                      │  Builds IVF index            │
                                      │  Runs search benchmarks      │
                                      └──────────────────────────────┘
```
