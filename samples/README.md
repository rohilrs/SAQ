# SAQ Samples

End-to-end examples demonstrating SAQ library usage.

## Available Samples

### saq_dbpedia_sample

Complete benchmark on the DBpedia 100K dataset (1536d OpenAI embeddings).

**Features:**
- Loads PCA-transformed vectors, centroids, cluster assignments, and ground truth from `.fvecs`/`.ivecs` files
- Builds SAQ-IVF index with per-segment CAQ encoding and 3-stage search
- Benchmarks search with varying nprobe (1-500)
- Computes recall@1, recall@10, recall@100, QPS
- Writes detailed results to file

**Prerequisites:**

Run Python preprocessing first (requires `faiss-cpu`, `numpy`):

```bash
cd python
python -m preprocessing.pca --data-dir ../data/datasets/dbpedia_100k
python -m preprocessing.ivf --data-dir ../data/datasets/dbpedia_100k -K 4096
python -m preprocessing.compute_gt --data-dir ../data/datasets/dbpedia_100k
```

This produces: `vectors_pca.fvecs`, `queries_pca.fvecs`, `centroids_4096_pca.fvecs`, `cluster_ids_4096.ivecs`, `variances_pca.fvecs`, `groundtruth.ivecs`

**Usage:**

```bash
# Build
cmake -B build -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build build

# Run benchmark
./build/samples/saq_dbpedia_sample <data_dir> <results_dir> <bpd> <K> <nprobe> <threads>

# Example: 4 bpd (8x compression), 4096 clusters, nprobe=200, 8 threads
./build/samples/saq_dbpedia_sample data/datasets/dbpedia_100k results/saq 4.0 4096 200 8
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `data_dir` | `data/datasets/dbpedia_100k` | Path to preprocessed dataset |
| `results_dir` | `results/saq` | Output directory for results |
| `bpd` | `2.0` | Bits per dimension (1.0, 2.0, 4.0, etc.) |
| `num_clusters` | `4096` | K for IVF clustering (must match preprocessing) |
| `nprobe` | `200` | Primary nprobe for search |
| `num_threads` | `8` | Thread count for index construction |

**Benchmark Results (DBpedia 100K, 1536D, K=4096, nprobe=200):**

| bpd | Compression | R@1 | R@10 | R@100 |
|-----|-------------|-----|------|-------|
| 1.0 | 32x | 85.0% | 87.3% | 86.6% |
| 2.0 | 16x | 92.8% | 92.6% | 90.0% |
| 4.0 | 8x | 97.0% | 94.8% | 90.9% |

## Building Samples

Samples are built when `SAQ_BUILD_SAMPLES=ON` (default):

```bash
cmake -B build -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build build
```

## OpenMP

With `SAQ_USE_OPENMP=ON`, index construction uses multiple threads. Control thread count with `OMP_NUM_THREADS`:

```bash
OMP_NUM_THREADS=8 ./build/samples/saq_dbpedia_sample ...
```
