# Optimal Cost Model Integration

## Goal

Replace SAQ's DP bit allocation cost model (`variance / 2^bits`) with precomputed optimal codebook MSE costs. This changes only which dimensions get how many bits — the encoder and search pipeline remain unchanged. Measure the impact on recall and cosine distortion.

## Background

The Python experiment on `gaussian_tail_codebook` showed that using actual DP-optimal codebook MSE as the cost signal changes bit allocation: high-variance (early PCA) blocks lose bits to lower-variance blocks. At 2 bpd this alone reduced total MSE by 38%. At 1 bpd it increased MSE by 27% (harmful reallocation at very low budgets). The combined effect with optimal codebooks was 5.5x lower MSE at 2 bpd.

This integration tests the allocation change in isolation with full end-to-end recall measurement.

## Changes

### 1. Python preprocessing: `python/preprocessing/compute_costs.py`

New script in the preprocessing pipeline, run after PCA:

```
python -m preprocessing.compute_costs --data-dir data/datasets/dbpedia_100k
```

- Loads `vectors_pca.fvecs` (N × D)
- For each dimension d (0..D-1):
  - Subsample 5000 values
  - Run `compute_codebook_dp()` (from experiments module) at bits 0..8
  - Record MSE at each bit rate
- Saves `optimal_costs.fvecs` as D × 9 matrix (row d = MSE at bits 0..8)
- Uses 2000 histogram bins (supports 8-bit codebooks)

### 2. C++ `SaqDataMaker` modifications

**`include/saq/quantization_plan.h`** — Add to `SaqDataMaker`:

```cpp
// New member:
FloatRowMat optimal_costs_;  // D × max_bits+1, or empty if not set

// New public method:
void set_optimal_costs(FloatRowMat costs) {
    optimal_costs_ = std::move(costs);
}
bool has_optimal_costs() const {
    return optimal_costs_.rows() > 0;
}
```

**`src/quantization_plan.cpp`** — In `dynamic_programming()`, replace cost computation:

```cpp
// Current (line 164):
auto v = var_sum / (1 << b);

// New:
double v;
if (has_optimal_costs()) {
    v = 0.0;
    size_t b_clamped = std::min(b, static_cast<size_t>(optimal_costs_.cols() - 1));
    for (size_t d = i * kDimPaddingSize; d < (i + j) * kDimPaddingSize; d++) {
        v += (d < static_cast<size_t>(optimal_costs_.rows()))
            ? optimal_costs_(d, b_clamped)
            : 0.0;
    }
} else {
    v = var_sum / (1 << b);
}
```

Also update the 0-bit tail cost (line 173-177) similarly.

### 3. IVF plumbing

**`include/index/ivf_index.h`** — Add method to `IVF`:

```cpp
void set_optimal_costs(FloatRowMat costs);
```

**`src/ivf_index.cpp`** — Forward to SaqDataMaker. Must be called before `construct()`.

### 4. Benchmark: `samples/saq_codebook_compare.cpp`

New sample that runs a controlled A/B comparison:

- Loads all data (vectors, queries, centroids, cluster_ids, ground truth, variances, optimal_costs)
- **Run A (baseline):** Build IVF with variance costs → search → record recall + timing
- **Run B (optimal costs):** Build IVF with optimal costs → search → record recall + timing
- **Cosine distortion:** For each query at nprobe=200:
  - Get top-K result IDs from both runs
  - Compute actual cosine similarity between query and each returned vector
  - Compare with ground truth top-K cosine similarities
  - Report: mean cosine distortion (difference from ground truth cosine)
- Print side-by-side table:
  ```
  nprobe | R@1 (A) | R@1 (B) | R@10(A) | R@10(B) | R@100(A) | R@100(B)
  ```
- Also print quantization plans from both runs to show bit allocation differences

### What doesn't change

- `caq_encoder.h` — still uniform quantization
- `caq_estimator.h`, `saq_estimator.h`, `saq_searcher.h` — same search pipeline
- `cluster_packer.h`, `cluster_data.h` — same storage format
- `lut.h`, `fast_scan.h` — same SIMD distance computation
- Index save/load — optimal costs are build-time only, not persisted in index

## Testing

- Existing benchmark (`saq_dbpedia_sample`) continues to work unchanged (no optimal costs → variance fallback)
- New benchmark (`saq_codebook_compare`) exercises the optimal cost path
- Python unit tests for `compute_costs.py` (correct output shape, costs decrease with more bits)

## Success criteria

- Build succeeds on MSVC + Ninja
- Baseline recall matches existing benchmarks (no regression)
- Quantization plans differ between A and B (confirms cost model is active)
- Recall comparison at 2 bpd and 4 bpd quantifies the allocation-only effect
