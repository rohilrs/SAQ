# SAQ Samples

End-to-end examples demonstrating SAQ library usage.

## Available Samples

### saq_dbpedia_sample

Complete benchmark on the DBpedia 100K dataset (1536d OpenAI embeddings).

**Features:**
- Loads vectors from .fvecs format
- K-means++ clustering for IVF partitioning
- Builds SAQ-IVF index with FastScan
- Benchmarks search with varying nprobe
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
  Clustering time: 6593.03 seconds

[3/5] Building SAQ-IVF index...
  Build time: 273.18 seconds
  FastScan enabled: yes

[4/5] Running search benchmarks...
  nprobe=  1 k=  1 recall=  8.00% QPS=  1284
  nprobe= 32 k=100 recall= 23.77% QPS=  1017
  ...

[5/5] Writing results...
Results written to: results/saq/dbpedia_100k_results.txt
```

## Building Samples

Samples are built when `SAQ_BUILD_SAMPLES=ON` (default):

```bash
mkdir build && cd build
cmake .. -DSAQ_BUILD_SAMPLES=ON
cmake --build .
```

## Configuration

The sample uses these default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_clusters` | 4 × √n | IVF partitions |
| `total_bits` | 64 | Bits per vector (768x compression for 1536d) |
| `num_segments` | 16 | Segments (4 bits each) |
| `use_fast_scan` | true | SIMD-accelerated scanning |
| `nprobe` | 1-128 | Clusters searched per query |

Modify these in `saq_dbpedia_sample.cpp` to experiment with different trade-offs.

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

See [python/README.md](../python/README.md) for more utilities.

## Results

Sample results are written to `results/saq/`:

```
results/saq/
└── dbpedia_100k_results.txt
```

Example output format:
```
================================================================================
SAQ-IVF Benchmark Results
================================================================================

Dataset Configuration:
  Base vectors:    99000
  Query vectors:   1000
  Dimension:       1536

Index Configuration:
  Clusters (K):    1258
  Total bits:      64
  FastScan:        enabled
  Compression:     768.0x

Search Results:
--------------------------------------------------------------------------------
  nprobe       k    Recall@k         QPS       Rel.Error    Search(ms)
--------------------------------------------------------------------------------
       1       1       8.00%      1284.0         0.3479        778.80
      ...
--------------------------------------------------------------------------------
```
