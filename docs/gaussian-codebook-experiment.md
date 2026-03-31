# Optimal Codebook vs Uniform Quantization: 2x2 Factorial Experiment

## Motivation

SAQ uses uniform scalar quantization per dimension: values are mapped to equally-spaced bins within [-v_max, v_max]. After PCA rotation, dimension values are approximately Gaussian, meaning uniform codebooks waste codes on sparse tails while under-resolving the dense center. This experiment measures the MSE improvement from replacing uniform codebooks with DP-optimal codebooks, and from using the resulting accurate MSE costs in SAQ's bit allocation DP.

## Method

### Optimal Codebook via DP

For each dimension, we find the optimal 1D quantizer by dynamic programming on sorted values:
- Bin data into a 500-bin weighted histogram
- `dp[j][i]` = min SSE for clustering bins 0..i into j contiguous groups
- Cost per range computed in O(1) via prefix sums: `SSE(a,b) = Σx² - (Σx)²/n`
- Backtrack to recover cluster centroids (the codebook)

This is equivalent to exact 1D k-means (which always produces contiguous clusters).

### 2x2 Factorial Design

Two independent variables:

| | Uniform Codebook | Optimal Codebook |
|---|---|---|
| **Variance cost** (`var/2^b`) | **A** — Baseline (current SAQ) | **B** — Codebook effect only |
| **Optimal cost** (actual MSE) | **C** — Allocation effect only | **D** — Combined |

For each condition:
1. Allocate bits per block using that condition's cost model (greedy allocation)
2. Compute total reconstruction MSE using that condition's codebook type
3. Normalize to MSE per dimension

### Dataset

DBpedia 100K (OpenAI text-embedding-3-large, 1536D), PCA-rotated. Experiment uses 10K vectors, first 192 dimensions (6 blocks of 32), max 6 bits/dim, 5K subsamples for DP.

## Results

### Total Reconstruction MSE

| bpd | A (baseline) | B (codebook) | C (allocation) | D (combined) |
|-----|-------------|-------------|---------------|-------------|
| 1.0 | 0.00196 | 0.00105 | 0.00248 | **0.00094** |
| 2.0 | 0.00171 | 0.00037 | 0.00106 | **0.00031** |
| 4.0 | 0.00008 | 0.00003 | 0.00006 | **0.00003** |

### Relative Improvement vs Baseline

| bpd | B/A (codebook) | C/A (allocation) | D/A (combined) | Interaction |
|-----|---------------|-----------------|---------------|-------------|
| 1.0 | 0.534 (-47%) | 1.266 (+27% worse) | 0.481 (-52%) | 1.40 (synergy) |
| 2.0 | 0.214 (-79%) | 0.620 (-38%) | 0.182 (-82%) | 0.73 |
| 4.0 | 0.410 (-59%) | 0.728 (-27%) | 0.316 (-68%) | 0.94 |

### Bit Allocation Shifts (2 bpd example)

| Block | Variance | A (baseline) | D (combined) | Change |
|-------|----------|-------------|-------------|--------|
| 0 | 0.2531 | 4 bits | 3 bits | -1 |
| 1 | 0.1271 | 3 bits | 2 bits | -1 |
| 2 | 0.0852 | 2 bits | 2 bits | — |
| 3 | 0.0617 | 1 bit | 2 bits | +1 |
| 4 | 0.0476 | 1 bit | 2 bits | +1 |
| 5 | 0.0381 | 1 bit | 1 bit | — |

### Per-Dimension Codebook Advantage (Uniform/Optimal MSE ratio)

| Bits | Min | p25 | Median | p75 | Max |
|------|-----|-----|--------|-----|-----|
| 1 | 1.53 | 4.37 | 4.94 | 5.69 | 9.09 |
| 2 | 1.70 | 2.59 | 2.85 | 3.18 | 4.71 |
| 4 | 1.42 | 2.04 | 2.21 | 2.40 | 3.18 |

Kurtosis correlates with improvement (r=0.46): more peaked (leptokurtic) dimensions benefit more from optimal codebooks.

## Findings

1. **Codebook effect dominates**: Optimal codebooks alone reduce MSE by 47-79% across bit rates. The largest improvement is at 2 bpd (79%), which is SAQ's primary operating point.

2. **Allocation effect is nuanced**: At 1 bpd, the optimal cost model *increases* MSE by 27% because it aggressively shifts bits away from high-variance blocks that still need them at very low budgets. At 2-4 bpd it helps (27-38% reduction).

3. **Combined effect shows synergy at 1 bpd** (interaction=1.40): the codebook improvement makes the reallocation viable by ensuring the blocks that lose bits still have good codebooks. At 2-4 bpd there is slight interference (the improvements overlap).

4. **Improvements exceed Lloyd-Max theory** (~1.42x for Gaussian): SAQ's per-vector v_max creates a data-dependent range that compounds with uniform spacing. The optimal codebook eliminates both issues simultaneously.

5. **Bit allocation redistribution**: The variance cost model (`var/2^b`) overestimates the benefit of extra bits for high-variance (early PCA) dimensions. Optimal costs redistribute bits toward lower-variance dimensions that benefit more from additional resolution.

## Reproduction

```bash
cd python/
python -m pytest experiments/test_compute_codebook.py -v  # 28 tests, ~5s
python -m experiments.compute_codebook --data-dir /path/to/dbpedia_100k  # ~60s
```

## Next Steps

- Integrate DP-optimal codebooks into C++ SAQ encoder (replace uniform quantization in `caq_encoder.h`)
- Evaluate impact on end-to-end recall (not just MSE)
- Test interaction with code adjustment (CAQ) — may become unnecessary with optimal codebooks
- Measure at full dimensionality (1536D) and with IVF cluster residuals
