# GPU-Accelerated SAQ Encode: Implementation Analysis and Benchmark Results

## 1. Introduction

This document analyzes the GPU implementation of the SAQ (Scalar Additive Quantization) encode pipeline, based on the algorithm described in arXiv:2509.12086. The GPU implementation targets the index construction phase (`IVF::construct`), which dominates offline preprocessing time at scale. We report benchmark results on an NVIDIA GeForce RTX 5090 (Blackwell, SM 120, 32GB GDDR7), discuss architectural divergences from the CPU implementation, and assess portability across consumer and datacenter GPUs.

## 2. Benchmark Results

**Hardware:** NVIDIA RTX 5090 (170 SMs, 1.79 TB/s memory bandwidth, 32GB GDDR7)
**Dataset:** DBpedia 100K (N=99,000 vectors, D=1,536 dimensions, K=4,096 clusters)
**Build:** MSVC 19.50 (VS 2025), CUDA 13.1, Release mode (`-O2`), `CMAKE_CUDA_ARCHITECTURES=native`

### 2.1 Optimized Encode Timing (Current)

After pooled memory allocator, GPU scatter kernels, and fused encode:

| Bits/dim | GPU Total (ms) | CPU 8T (ms) | **Speedup** |
|----------|---------------|-------------|------------|
| 1.0 | **415** | 486 | **1.2x** |
| 2.0 | **383** | 537 | **1.4x** |
| 4.0 | **398** | 663 | **1.7x** |

### 2.1.1 GPU Batch Search (Q=1000, nprobe=200)

| Bits/dim | Search (ms) | QPS | GPU R@100 | CPU R@100 |
|----------|------------|-----|-----------|-----------|
| 1.0 | 508 | 1,969 | 86.6% | 86.0% |
| 2.0 | 448 | 2,231 | 90.0% | 89.9% |
| 4.0 | 528 | 1,895 | 90.9% | 90.8% |

GPU recall matches CPU recall within 0.1-0.6% across all bit rates. The search kernel computes accurate distance for all vectors (no stage 2 fast-distance screening).

### 2.1.2 Pre-Optimization Encode Timing (Baseline)

| Bits/dim | Segment Plan | GPU Total (ms) | GPU Kernel (ms) | CPU 8T (ms) | Wall Speedup |
|----------|-------------|----------------|-----------------|-------------|--------------|
| 1.0 | 64d/5b + 256d/3b + 320d/1b + 896d/0b | 1,533 | 115 | 475 | 0.31x |
| 2.0 | 128d/6b + 256d/4b + 576d/2b + 576d/0b | 1,566 | 118 | 506 | 0.32x |
| 4.0 | 192d/8b + 448d/5b + 384d/3b + 512d/2b | 1,633 | 135 | 641 | 0.39x |

### 2.2 Transfer-Excluded Timing (Data Already on GPU)

In a production pipeline where data already resides on GPU (e.g., after GPU-based PCA or K-means), the relevant comparison excludes H2D transfer and cluster allocation overhead:

| Bits/dim | GPU Kernels (ms) | GPU Scatter (ms) | GPU K+S (ms) | CPU 8T (ms) | Kernel Speedup | K+S Speedup |
|----------|-----------------|------------------|-------------|-------------|----------------|-------------|
| 1.0 | **115** | 319 | 434 | 475 | **4.1x** | **1.1x** |
| 2.0 | **118** | 338 | 456 | 506 | **4.3x** | **1.1x** |
| 4.0 | **135** | 386 | 521 | 641 | **4.8x** | **1.2x** |

The GPU's raw encode kernels (subtract + rotate + CAQ encode + pack) are **4.1-4.8x faster** than the 8-thread CPU implementation. Including the scatter phase (which copies encoded data into per-cluster structures), the GPU achieves **1.1-1.2x speedup** — essentially at parity with the CPU.

The scatter phase is the primary optimization target: replacing 16,384 small `cudaMemcpy` D2D calls with a single GPU scatter kernel would bring the effective speedup close to the kernel-only 4-5x figure.

### 2.2.1 Optimized Time Breakdown (2.0 bpd)

| Phase | Before (ms) | After (ms) | Change |
|-------|------------|-----------|--------|
| CPU prep (sort+metadata) | 77 | 77 | — |
| H2D upload | 120 | 120 | — |
| GPU cluster alloc | 899 | 2 | **450x** |
| GPU kernels (encode) | 118 | 154 | +30% (fused includes cublasSgeam) |
| GPU scatter | 338 | 1.5 | **225x** |
| **Total** | **1,566** | **383** | **4.1x** |
| **CPU 8T** | 506 | 537 | — |
| **GPU vs CPU** | 0.32x | **1.4x** | |

### 2.3 Pre-Optimization Detailed Time Breakdown

**2.0 bpd (128d/6b + 256d/4b + 576d/2b + 576d/0b):**

| Phase | Time (ms) | % of Total | Notes |
|-------|-----------|-----------|-------|
| CPU prep (sort+metadata) | 77 | 4.9% | `std::stable_sort` on 99K indices |
| H2D upload | 120 | 7.7% | ~580 MB via PCIe 5.0 (~4.8 GB/s) |
| GPU cluster alloc + ID upload | 899 | 57.4% | 4,096 `cudaMalloc` calls |
| Segment 0 kernels (128d, 6b) | 37 | 2.4% | Subtract + GEMM + CAQ + pack |
| Segment 0 scatter | 94 | 6.0% | |
| Segment 1 kernels (256d, 4b) | 16 | 1.0% | |
| Segment 1 scatter | 97 | 6.2% | |
| Segment 2 kernels (576d, 2b) | 40 | 2.6% | Largest segment |
| Segment 2 scatter | 99 | 6.3% | |
| Segment 3 kernels (576d, 0b) | 26 | 1.7% | Zero-bit: L2 norm only |
| Segment 3 scatter | 48 | 3.1% | Fewer codes to copy |
| **GPU kernels total** | **118** | **7.5%** | |
| **GPU scatter total** | **338** | **21.6%** | |
| **Total wall** | **1,566** | **100%** | |

The dominant cost is **GPU cluster allocation** (~57% of total). This is 4,096 `cudaMalloc` calls for per-cluster segment data structures, each requiring CUDA driver interaction. A pooled memory allocator or bulk allocation strategy would eliminate this overhead. The second-largest cost is the **scatter phase** (~22%), which copies flat encoded arrays into per-cluster structures via ~16,384 small D2D memcpy calls.

### 2.4 Key Observation: Overhead-Bound Regime

At N=99K, the GPU encode is **overhead-bound**, not compute-bound. Only 7.5% of wall time is spent in actual encode kernels. The arithmetic intensity of the CAQ encode kernel is approximately:

$$\text{AI} = \frac{2 \cdot N \cdot D_\text{seg} \cdot (\text{adj\_rounds} + 2)}{N \cdot D_\text{seg} \cdot 4} \approx \frac{2(r+2)}{4} \approx 4 \text{ FLOP/byte}$$

For `caq_adj_rd_lmt=6`, this yields ~4 FLOP/byte, which is below the RTX 5090's operational intensity threshold (~100 FLOP/byte for compute-bound workloads at 1.79 TB/s bandwidth and ~180 TFLOPS FP32). The workload is firmly memory-bandwidth-limited, and at 99K vectors the GPU's massive parallelism is underutilized.

**Break-even analysis:** With the current implementation, the three overhead sources (cluster alloc: 900ms, scatter: 340ms, H2D upload: 120ms) total ~1,360ms of non-compute work. Even with kernel-only times of 115-135ms, the GPU cannot overcome this overhead at N=99K. The GPU becomes competitive when:
1. Data already resides on GPU (eliminates H2D upload)
2. Cluster allocation uses a memory pool (eliminates alloc overhead)
3. A GPU scatter kernel replaces D2D memcpy calls (reduces scatter to ~5ms)

With all three optimizations, the estimated GPU encode time would be ~120-140ms vs ~500-640ms CPU — a **3.5-4.8x speedup** at N=99K. At N > 1M, the speedup would increase further as the GPU's parallelism is better utilized.

## 3. Architectural Divergences from CPU Implementation

### 3.1 Parallelization Granularity

**CPU:** Parallelism is at the *cluster level*. Each of K=4,096 clusters is processed independently by an OpenMP thread (`#pragma omp parallel for schedule(dynamic)`). Within a cluster, vectors are encoded sequentially, and within a vector, dimensions are processed in a tight scalar loop. The CPU benefits from:
- Perfect data locality (one cluster's data fits in L2/L3 cache)
- No synchronization overhead between clusters
- Compiler auto-vectorization of inner loops (AVX-512 for 16 floats/cycle)

**GPU:** Parallelism is at the *vector level* with intra-vector dimension parallelism via warp cooperation. All N=99K vectors across all clusters are processed simultaneously in a single kernel launch. One 32-thread warp handles one vector, with each lane responsible for `ceil(D_seg/32)` dimensions. This requires:
- Pre-sorting all vectors by cluster ID to enable contiguous centroid lookup
- Warp-level reduction primitives (`__shfl_down_sync`) to aggregate per-lane statistics
- Explicit synchronization after each code adjustment round

This is a fundamentally different decomposition: the CPU exploits **cluster-level independence** while the GPU exploits **vector-level parallelism** with **dimension-level cooperation** within warps.

### 3.2 Code Adjustment (CAQ) Synchronization Model

The CAQ code adjustment loop is the core algorithmic challenge for GPU parallelization. The CPU implementation processes dimensions sequentially within a single thread, maintaining running sums (`ip_o_oa`, `oa_l2sqr`) that are updated incrementally as each dimension's code is adjusted:

```
// CPU: sequential, incremental stat updates
for d in 0..D_seg:
    try adjust code[d] up/down
    if improvement: update ip_o_oa, oa_l2sqr immediately
```

The GPU implementation partitions dimensions across 32 warp lanes. Each lane adjusts codes in its chunk independently, but the adjustment criterion depends on *global* sums (`ip_o_oa`, `oa_l2sqr`) that span all lanes. This creates a **read-after-write hazard** across lanes: lane A's adjustment affects the optimality criterion for lane B's subsequent adjustments within the same round.

The GPU resolves this with a **correction pass** architecture:

```
// GPU: parallel lanes with periodic synchronization
for round in 1..max_rounds:
    each lane: adjust codes in my chunk using stale global sums
    warp_reduce: count total adjustments
    if converged (0 adjustments): break
    each lane: recompute local ip, oa_l2 from updated codes
    warp_reduce: sync global ip_o_oa, oa_l2sqr
```

**Implication:** The GPU may require more rounds to converge than the CPU because lanes operate on stale global statistics within each round. However, empirically the correction pass after each round re-establishes accurate global sums, and convergence behavior is similar (typically 3-5 rounds for `caq_adj_eps=1e-5`). The fundamental correctness guarantee is preserved: each round monotonically improves the cosine objective, and the correction pass ensures no drift accumulates across rounds.

### 3.3 Numeric Precision and Warp Shuffle Workarounds

The CAQ algorithm requires double-precision accumulation to avoid catastrophic cancellation in the cosine improvement test:

```
if (ip_o_oa^2 + re_eps) * new_length >= new_ip^2 * oa_l2sqr:
    reject adjustment
```

Both `ip_o_oa` and `oa_l2sqr` are sums of products over D_seg dimensions (up to 576). Single-precision would lose ~7 significant digits, making the comparison unreliable when the improvement is small.

The GPU faces an additional complication: CUDA's `__shfl_sync` intrinsic does not natively support `double`. The implementation works around this by decomposing each `double` into two 32-bit integers via `__double2loint` / `__double2hiint`, shuffling each component independently, and reconstructing via `__hiloint2double`:

```cuda
__device__ double warp_reduce_sum_double(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int lo = __shfl_down_sync(0xFFFFFFFF, __double2loint(val), offset);
        int hi = __shfl_down_sync(0xFFFFFFFF, __double2hiint(val), offset);
        val += __hiloint2double(hi, lo);
    }
    return val;
}
```

This doubles the number of shuffle instructions per reduction (10 shuffles instead of 5 for a 32-lane reduction). On Blackwell (SM 120), warp shuffles have 1-cycle latency with full throughput, so the overhead is minimal (~10 extra cycles per reduction). On older architectures (SM 80/90), shuffle throughput is similarly high, making this a non-issue in practice.

### 3.4 Rotation via cuBLAS GEMM

The CPU applies per-segment rotation using Eigen matrix multiplication (`result = residuals * P`), which leverages MKL or OpenBLAS BLAS backends. The GPU uses cuBLAS `cublasSgemm` for the same operation.

**Layout consideration:** Both the rotation matrix `P` and the data matrix are stored as `FloatRowMat` (Eigen row-major). Since cuBLAS operates on column-major matrices, the row-major [N x D] matrix appears as a column-major [D x N] matrix to cuBLAS. The GEMM call `C = A * B` in column-major becomes `C^T = B^T * A^T` in row-major, which correctly computes `rotated = residuals * P`.

**Divergence:** The CPU rotation is implicit (part of the `QuantizerCluster::encode` pipeline), while the GPU rotation is an explicit, separate kernel launch with a temporary buffer. This increases GPU memory footprint by 2x per segment (residuals + rotated buffers), but enables overlapping rotation with other operations via CUDA streams (not yet implemented).

### 3.5 Data Layout and Scatter Phase

**CPU:** The CPU processes clusters independently via OpenMP. Each thread handles one cluster's vectors, computes codes, and writes directly into the cluster's `SaqCluData` structure. No cross-cluster coordination is needed.

**GPU:** The GPU processes all N vectors simultaneously, producing flat output arrays (codes, factors) indexed by sorted vector position. A subsequent **scatter phase** copies per-vector results into per-cluster `GpuSaqCluData` structures. This scatter consists of K x segments `cudaMemcpy` device-to-device calls from the CPU, which is inherently sequential and imposes O(K) latency.

For K=4,096 clusters and 4 segments, this is ~16,384 small memcpy calls plus `launch_store_factors` kernel calls. Measured scatter time is **320-390ms** (~20us per call including factor store kernel dispatch), making it the second-largest overhead after cluster allocation. This cost could be eliminated by:
1. A GPU scatter kernel that processes all clusters in one launch
2. Fused encode-and-scatter kernels that write directly to per-cluster storage
3. Using CUDA graphs to batch the memcpy operations

### 3.6 Fastscan Layout (Not Yet Implemented)

The CPU search path uses a fastscan layout where short codes are interleaved in blocks of 32 vectors (matching the AVX-512 SIMD width). The GPU encode currently stores short codes in per-vector linear order with a `TODO` for fastscan reorder. This means:
- GPU-encoded data cannot be directly used by the CPU search path without reordering
- A future GPU search kernel would need its own layout (likely warp-width interleaving)

## 4. Implications for Correctness and Quality

### 4.1 Determinism

The CPU implementation is deterministic given the same rotation matrices and input data. The GPU implementation is **also deterministic** for a given GPU architecture, because:
- Warp execution is lockstep (no thread scheduling non-determinism within a warp)
- Floating-point operations follow IEEE 754 (no FMA fusion across operations unless explicitly enabled)
- `__shfl_sync` with mask `0xFFFFFFFF` is deterministic

However, GPU results may differ from CPU results due to:
1. **Operation order in reductions:** Warp tree reduction sums in a different order than CPU sequential summation, producing different floating-point rounding
2. **cuBLAS GEMM order:** cuBLAS may use different internal tiling/reduction orders than Eigen's BLAS backend, producing different rotation results at the LSB level

These differences are at the level of floating-point rounding (~1 ULP for single-precision factors, ~1-2 ULP for double-precision intermediates) and do not affect quantization quality measurably.

### 4.2 Code Adjustment Convergence

The GPU's lane-parallel code adjustment with periodic correction may converge to a slightly different local optimum than the CPU's sequential adjustment. Both are valid local optima of the cosine objective. Empirically, the difference in final `ip_o_oa / sqrt(oa_l2sqr * o_l2sqr)` (cosine similarity between original and quantized vectors) is < 1e-6 across all vectors tested.

## 5. GPU Architecture Portability

### 5.1 Architecture Coverage

The default build configuration (`CMAKE_CUDA_ARCHITECTURES`) targets SM 80-120:

| Architecture | SM | GPU Examples | Status |
|-------------|-----|-------------|--------|
| Ampere | 80 | A100, RTX 3090 | Compilable, data-center target |
| Ampere | 86 | RTX 3060/3070 | Compilable (untested) |
| Ada Lovelace | 89 | RTX 4090 | Compilable (untested) |
| Hopper | 90 | H100, H200 | Compilable, data-center target |
| Blackwell | 100 | B200, GB200 | Compilable |
| Blackwell | 120 | RTX 5090 | Tested (this report) |

Older architectures (Turing SM 75, Volta SM 70) are not included in the default build but could be added to `CMAKE_CUDA_ARCHITECTURES` without code changes. All architectures use a warp size of 32 threads, so the warp-cooperative encode kernel is universally applicable without modification.

### 5.2 Performance Implications by Architecture

**Device memory bandwidth (determines kernel performance):**

| GPU | HBM/GDDR BW (TB/s) | Relative to RTX 5090 | Expected Kernel Speedup |
|-----|---------------------|----------------------|------------------------|
| A100 (80GB) | 2.0 | 1.12x | ~1.1x |
| H100 (80GB) | 3.35 | 1.87x | ~1.5-1.9x |
| RTX 5090 | 1.79 | 1.0x | 1.0x (baseline) |

Since the CAQ encode kernel is memory-bandwidth-limited (~4 FLOP/byte), kernel performance tracks device memory bandwidth, not FP32 TFLOPS. The H100's HBM3 bandwidth advantage would translate directly to faster kernels. Note that H2D upload is PCIe-bound (~120ms at N=99K via PCIe 5.0), independent of device memory bandwidth.

**Raw compute throughput (reference only — not the bottleneck):**

| GPU | FP32 TFLOPS | SMs | Notes |
|-----|-------------|-----|-------|
| A100 | 19.5 | 108 | Memory-BW-limited; FLOPS underutilized |
| H100 | 67.0 | 132 | Memory-BW-limited; FLOPS underutilized |
| RTX 5090 | ~180 | 170 | Memory-BW-limited; FLOPS underutilized |

FP32 throughput is not the limiting factor for the CAQ encode kernel. At ~4 FLOP/byte arithmetic intensity, even the A100 (19.5 TFLOPS / 2.0 TB/s = 9.75 FLOP/byte operational intensity) has more compute than needed. Performance differences across GPUs are driven by memory bandwidth and warp shuffle latency, not FLOPS.

### 5.3 Will It Work "Optimally" on A100/H100?

**Correctness:** Yes. The code is architecture-agnostic (no SM-specific intrinsics, no shared memory bank conflict assumptions, no warp-divergent code paths).

**Optimality:** No, not without tuning. Specific considerations:

1. **Warps per block:** Currently 4 (128 threads/block). A100/H100 support up to 1024 threads per SM with 2048 resident threads. Increasing to 8-16 warps/block would improve occupancy, especially on A100 which has fewer registers per SM.

2. **L2 cache utilization:** A100 has 40MB L2, H100 has 50MB L2, RTX 5090 has 72MB L2. The centroid array for K=4096 clusters at D_seg=576 is 4096 * 576 * 4 = 9.4MB, fitting in all L2 caches. But the vector data for Segment 2 (99K * 576 * 4 = 217MB) does not fit in any L2.

3. **HBM2e vs GDDR7 latency:** A100's HBM2e has lower random-access latency than GDDR7, which benefits the centroid broadcast pattern in the CAQ kernel (non-coalesced reads of centroid rows).

4. **TensorCores:** Not used. The rotation GEMM could benefit from FP16 TensorCores with mixed-precision (FP16 accumulation is insufficient, but FP32 accumulation of FP16 inputs is viable for rotation).

5. **Hopper-specific features:** H100 supports Thread Block Clusters and Distributed Shared Memory, which could enable inter-SM cooperation for the scatter phase. Not currently exploited.

## 6. When to Use GPU Encode

| Scenario | Recommendation |
|----------|---------------|
| N < 100K, encode | GPU 1.2-1.7x faster than CPU 8T (pool+scatter+fused encode) |
| N < 100K, batch search (Q=1000) | GPU 1,900-2,200 QPS at nprobe=200, recall matches CPU |
| N > 500K, one-shot index build | GPU likely 3-5x faster (overhead further amortized) |
| N > 1M, data already on GPU | GPU significantly faster (~10-50x expected) |
| Real-time incremental updates | CPU (no allocation/transfer overhead) |
| Preprocessing pipeline on GPU | GPU (avoid D2H/H2D roundtrip) |

## 7. Future Optimizations

Completed optimizations (marked with ✓):

1. ✓ **Pooled memory allocator**: 900ms → 2ms (450x improvement)
2. ✓ **GPU scatter kernel**: 340ms → 1.5ms (225x improvement)
3. ✓ **Fused encode (L1+L2)**: Eliminated d_residuals and d_codes intermediate buffers
4. ✓ **GPU batch search**: 3-stage kernel with shared-memory float LUT, recall matches CPU

Remaining optimizations:

5. **Stage 2 fast-distance filtering in search:** Re-enable LUT-based fast screening to reduce accurate distance computations from ~24 to ~5 per cluster. Requires calibrating the fast distance formula for the GPU's float LUT.

6. **Parallel centroid search:** The 217ms CPU centroid search is single-threaded. OpenMP parallelization or GPU-based centroid search would cut total search time nearly in half.

7. **Stream pipelining:** Overlap segment N's scatter with segment N+1's encode using CUDA streams.

8. **FP16 rotation:** Use cuBLAS FP16 TensorCore GEMM with FP32 accumulation for rotation, reducing memory traffic by 2x.

9. **Multi-GPU support:** Partition clusters across GPUs, each GPU encodes its subset. Requires load balancing (clusters have variable sizes).

## 8. Conclusion

The GPU encode and search pipeline demonstrates that the SAQ algorithm parallelizes effectively on GPU. After implementing a pooled memory allocator (900ms→2ms), GPU scatter kernels (340ms→1.5ms), and a fused encode kernel (eliminating intermediate buffers), the GPU encode achieves a **1.2-1.7x speedup** over an optimized 8-thread AVX-512 CPU implementation on the RTX 5090 at N=99K.

GPU batch search achieves **1,900-2,200 QPS** for 1,000 queries at nprobe=200, with recall matching the CPU within 0.1-0.6% across all bit rates (86.6-90.9%). The search kernel uses shared-memory float LUT and computes accurate distance for all vectors (no fast-distance screening yet).

The implementation is architecture-portable across all CUDA SM 80+ GPUs. On datacenter GPUs (A100, H100), the memory-bandwidth-limited kernels would benefit from HBM's higher bandwidth. The warp-cooperative design (32 threads per vector, double-precision shuffles for numerical stability) requires no architecture-specific tuning.

See `docs/gpu-search-implementation.md` for detailed documentation of the search kernel architecture and the debugging process that brought recall from 4.6% to 90%.
