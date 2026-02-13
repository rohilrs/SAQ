# SAQ Samples

End-to-end examples demonstrating SAQ library usage.

## Available Samples

### saq_dbpedia_sample

Complete benchmark on the DBpedia 100K dataset (1536d OpenAI embeddings).

**Features:**
- Loads vectors from .fvecs format
- K-means++ clustering for IVF partitioning
- Builds SAQ-IVF index with scalar quantization
- Benchmarks search with varying nprobe (1-128)
- Computes recall@k, QPS, and relative error
- Writes detailed results to file

**Usage:**
```bash
# 1. Download dataset (one-time)
python samples/download_dbpedia.py

# 2. Build sample
cd build
cmake --build . --target saq_dbpedia_sample

# 3. Run (from project root)
cd ..
./build/samples/saq_dbpedia_sample

# Or with custom paths:
./build/samples/saq_dbpedia_sample <data_dir> <results_dir>
```

**Output:**
```
================================================================================
SAQ-IVF Sample: DBpedia 100K Dataset
================================================================================

[1/5] Loading data...
  Base vectors: 99000 x 1536
  Queries: 1000 x 1536
  Ground truth: 1000 queries

[2/5] Clustering (1258 clusters)...
  Clustering time: ...

[3/5] Building SAQ-IVF index...
  Build time: ...

[4/5] Running search benchmarks...
  nprobe=  1 k=  1 recall=  ...% QPS= ...
  nprobe= 32 k=100 recall=  ...% QPS= ...

[5/5] Writing results...
Results written to: results/saq/dbpedia_100k_results.txt
```

## Building Samples

Samples are built when `SAQ_BUILD_SAMPLES=ON` (default):

```bash
mkdir build && cd build
cmake .. -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON
cmake --build .
```

## Configuration

The sample uses these default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_clusters` | 4 x sqrt(n) | IVF partitions |
| `total_bits` | 64 | Bits per vector (768x compression for 1536d) |
| `nprobe` | 1-128 | Clusters searched per query (swept) |
| `k` | 1, 10, 100 | Nearest neighbors returned (swept) |

Modify these in `saq_dbpedia_sample.cpp` to experiment with different trade-offs.

## OpenMP

With `SAQ_USE_OPENMP=ON`, the index build and search use multiple threads. Control thread count with `OMP_NUM_THREADS`:

```bash
OMP_NUM_THREADS=8 ./build/samples/saq_dbpedia_sample
```

## Adding New Samples

1. Create `samples/your_sample.cpp`
2. Add to `samples/CMakeLists.txt`:

```cmake
add_executable(your_sample
  ${CMAKE_CURRENT_SOURCE_DIR}/your_sample.cpp
)
target_link_libraries(your_sample PRIVATE saq)
```

3. Rebuild

## Python Scripts

| Script | Purpose |
|--------|---------|
| `download_dbpedia.py` | Download DBpedia 100K from HuggingFace |

See [python/README.md](../python/README.md) for more utilities and the Python benchmark.

## Results

Sample results are written to `results/saq/`:

```
results/saq/
└── dbpedia_100k_results.txt
```
