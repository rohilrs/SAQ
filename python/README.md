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

**Status: Needs updating.** The bindings in `bindings/saq_bindings.cpp` reference the old C++ API (`SAQQuantizer`, `IVFIndex`, etc.) which was replaced during the reference alignment refactor. The current C++ API uses `SAQuantizer`, `IVF`, `QuantizeConfig`, `SearcherConfig`, etc.

To use SAQ from Python, use the preprocessing scripts above and run the C++ benchmark directly.

### Building (when bindings are updated)

```bash
cmake -B build -DSAQ_BUILD_PYTHON=ON
cmake --build build --target _saq_core
```

## Utility Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `preprocessing/pca.py` | `preprocessing/` | PCA via Faiss PCAMatrix |
| `preprocessing/ivf.py` | `preprocessing/` | K-means via Faiss |
| `preprocessing/compute_gt.py` | `preprocessing/` | Brute-force ground truth |
| `preprocessing/utils/io.py` | `preprocessing/utils/` | Read/write fvecs/ivecs/fbin/ibin |
| `ivf_clustering.py` | root | Standalone K-means clustering utility |
| `benchmark_saq.py` | root | Python benchmark (requires updated bindings) |

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
