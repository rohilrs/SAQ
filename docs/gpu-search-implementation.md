# GPU Batch Search: Implementation, Debugging, and Lessons Learned

**Date:** 2026-03-16
**Branch:** `gpu`

## 1. Overview

This document covers the complete implementation of GPU-accelerated batch search for the SAQ (Scalar Additive Quantization) index, including the encode-side optimizations that preceded it, the search kernel architecture, and the multi-day debugging effort that brought recall from 4.6% to 90%.

### Final Results

| Metric | Before | After |
|--------|--------|-------|
| GPU Encode (2 bpd) | 1,566ms | 347ms (1.5x faster than CPU 8T) |
| GPU Search (Q=1000, nprobe=200) | N/A | 368ms total, 2,716 QPS |
| GPU Recall@100 | N/A | 89.95% (matches CPU 89.91%) |

## 2. Encode-Side Optimizations (Chunks 1-3)

### 2.1 Pooled Memory Allocator

**Problem:** 115,000 `cudaMalloc` calls (7 arrays × 4 segments × 4,096 clusters) consumed 900ms.

**Solution:** `GpuMemoryPool` performs 28 bulk allocations (one per array type per segment), computes prefix-sum offset tables, and assigns raw pointers into `GpuSaqCluData` views.

**Key design decisions:**
- Pool owns all memory via `DevicePtr<T>` (RAII). `GpuSaqCluData` becomes a lightweight view with non-owning pointers.
- Offset tables (`cluster_offsets`, `block_offsets`) are uploaded to device memory for use by scatter and search kernels.
- `assign_pointers()` fills per-cluster segment descriptors from the pool's contiguous allocations.

**Impact:** 900ms → 2ms allocation time.

### 2.2 GPU Scatter Kernels

**Problem:** Per-cluster `cudaMemcpy` D2D loop made ~16K API calls per segment, costing ~340ms.

**Solution:** Three CUDA kernels process all N vectors in parallel:
- `kernel_scatter_short_codes`: Reads 1-bit-per-dim linear packed format, regroups into 4-bit codebook indices in GPU blocked layout (1 byte per codebook per vector, organized in blocks of 32).
- `kernel_scatter_factors`: Writes `o_l2norm` and `ip_cent_oa` in blocked layout (blocks of 32), `rescale` and `error` in per-vector layout indexed by `cluster_offsets`.
- `kernel_scatter_long_codes`: Simple byte copy with offset computation.

**Key detail — codebook bit ordering:** The LUT's `kPos` array maps bit 3 → dim 0, bit 2 → dim 1, etc. (MSB of 4-bit index = first dimension). The scatter kernel must assemble `code4 |= (bit << (3 - j))` to match. Getting this reversed was the first recall bug (4.6% → 41%).

**Impact:** 340ms → 1.5ms scatter time.

### 2.3 Fused Encode Kernel (L1 + L2)

**L1 — Eliminate d_residuals:** Instead of a separate `subtract_centroid` kernel, the fused encode takes the GEMM output on raw vectors and subtracts the rotated centroid inline. The raw vector segment is extracted via `cublasSgeam`, rotated via `cublasSgemm`, and the rotated centroid (computed on CPU, uploaded) is subtracted per-warp in the encode kernel.

**L2 — Eliminate d_codes:** Integer codes stay in local register arrays (`local_codes[kMaxDimsPerLane]`) instead of global memory `d_codes`. Short and long codes are packed inline after the CAQ adjustment loop converges, using `atomicOr` for byte-boundary cases in short code packing.

**Key bug found (the rescale bug):** The fused encode was missing the `rescale_vmx_to1` step. The CPU encoder applies `fac_rescale *= v_mx` after computing `fac_rescale = o_l2sqr / ip_o_oa`. This normalization is critical — without it, `rescale ≈ 1.0` instead of `≈ 0.15`, causing the search distance formula to overestimate inner products by ~7x.

## 3. GPU Search Architecture

### 3.1 Host-Side Orchestration (`gpu_ivf_search.cpp`)

The search pipeline:
1. **Centroid search** (CPU, single-threaded): `FlatInitializer::centroids_distances` finds nprobe nearest centroids per query. (~217ms for Q=1000)
2. **Query rotation** (CPU, Eigen): Per segment, rotate query via `query * P_s`. Compute per-segment constants: `sum_q`, `q_l2sqr`, `sq_delta = 2/(1 << num_bits)`.
3. **Upload**: Rotated queries, centroid IDs, query constants, descriptor tables.
4. **Search kernel**: One block per (query, cluster) pair.
5. **Merge kernel**: One block per query, selects top-K from all cluster candidates.
6. **Download**: Result IDs.

### 3.2 Device-Side Descriptor Tables

The search kernel cannot dereference host pointers. Two descriptor structs marshal pool data to the GPU:

```cpp
struct GpuSegmentDescriptor {
    uint8_t* short_codes;       // pool base pointer for this segment
    uint8_t* long_codes;
    float* factor_o_l2norm;     // blocked layout
    float* factor_rescale;      // per-vector layout
    float* centroids;
    size_t num_codebooks;       // D_seg / 4
    size_t D_seg, num_bits, long_bytes_per_vec;
};

struct GpuClusterDescriptor {
    size_t num_vec, num_blocks;
    uint32_t* ids;              // pointer into pool.ids
};
```

These are constructed on the host from pool metadata, uploaded once per `search_batch` call.

### 3.3 Search Kernel Design

**Grid:** `dim3(Q, nprobe)` — one block per (query, cluster) pair.
**Block:** 128 threads (4 warps).

**Phase 0 — Build LUT in shared memory:**

For each segment, the kernel cooperatively computes the residual query `query_seg - centroid_seg` in shared memory, then builds a float LUT (16 entries per codebook, representing all subset sums of 4 query dimensions). This uses the same `LUT[j] = LUT[j - lowbit(j)] + query[kPos[j]]` recurrence as the CPU's `pack_lut`.

Shared memory layout: `[float LUT][per-segment constants][work counter][residual query]`. For D=1536 with 384 codebooks: ~24KB LUT + ~6KB residual + ~1KB overhead = ~31KB, well within 128KB/SM.

**Phase 1 — Accurate distance for all vectors:**

The current implementation skips the fast-distance screening (stage 2) and computes accurate distance for every valid vector. This guarantees correct recall at the cost of evaluating all ~24 vectors per cluster instead of the ~5 that would pass stage 2.

Each warp claims 32-vector blocks via work-stealing (`atomicAdd` on a shared counter). For each vector, the kernel computes:

```
For each segment s:
    lut_sum = sum over codebooks of LUT[short_code[cb]]   // shared memory lookup
    ext_ip = sum over dims of query_resid[d] * long_code_val[d]  // long code IP
    full_ip = lut_sum + ext_ip * sq_delta + (-1 + sq_delta/2) * sum_q
    ip_o_q = rescale * full_ip
    seg_dist = o_l2sqr + q_l2sqr - 2 * ip_o_q
total_dist = sum of seg_dist
```

Candidates are collected via lane-0 shuffle collection into a per-warp buffer (64 entries), with distk-based eviction when full.

**Phase 2 — Output:**

The 4 warps' candidates are merged into a per-block output buffer (256 entries max) using `atomicAdd` for position allocation. Written to global memory at `[q * nprobe + cluster_rank]` slot.

### 3.4 Merge Kernel

One block per query, single thread (sufficient for ~1K candidates). Loads all candidates from nprobe cluster slots, performs selection sort to find top-K, writes final result IDs.

### 3.5 Long Code IP Computation

The GPU packs long codes in a sequential per-dimension bit layout (different from the CPU's bit-plane interleaved format). The `gpu_long_code_ip` device function unpacks `(num_bits - 1)` bits per dimension from this layout and computes the dot product with the residual query. Both the encode packing and search unpacking use the same layout, maintaining self-consistency.

## 4. The Recall Debugging Journey

### 4.1 Initial State: 4.6% Recall

The first working search kernel produced 4.6% recall. The GPU found some correct vectors but ranked them poorly.

### 4.2 Bug 1: Codebook Bit Order (4.6% → 41%)

**Symptom:** LUT lookups returned wrong values for every vector.

**Root cause:** The scatter kernel assembled the 4-bit codebook index as `code4 |= (bit << j)` for `j=0..3`, putting dim 0 in bit 0. But the LUT's `kPos` array maps bit 3 → dim 0 (MSB = first dimension).

**Fix:** `code4 |= (bit << (3 - j))` — dim 0 goes to bit 3.

### 4.3 Debugging at 41%: Ruling Out Candidates

With 41% recall, the distance formula was partially correct but not accurate enough. Systematic debugging:

1. **All-vectors test:** Removed stage 2 filtering to compute accurate distance for ALL vectors. Recall unchanged at 41% → confirmed the issue is in the distance VALUES, not candidate filtering.

2. **Long codes disabled:** Setting long codes to 0 gave 36% recall. Re-enabling gave 41%. Long codes contributed only 5% instead of the expected ~50% improvement.

3. **Brute-force comparison:** CPU top-1 (vec 63099, bf_dist=0.57) appeared at GPU rank 7. GPU results contained nearby vectors (bf_dist 1.0-1.2) but missed the closest one.

4. **CPU vs GPU result overlap:** 39/100 overlap between GPU and CPU top-100.

### 4.4 Bug 2: Missing rescale_vmx_to1 (41% → 90%)

**Key diagnostic:** Downloaded GPU `rescale` factor for vec 63099 and compared with CPU:

| Factor | CPU | GPU |
|--------|-----|-----|
| `rescale` (seg 0) | 0.154 | 1.018 |
| `o_l2norm` (seg 0) | 0.583 | 0.583 |

The `o_l2norm` matched perfectly, but `rescale` differed by 6.6x.

**Red herrings investigated:**
- Floating-point non-commutativity: `(raw-cent)*P` vs `raw*P - cent*P` — tested by computing residuals on host and uploading. Didn't fix the issue.
- Different random rotations between CPU and GPU: Added `std::srand(42)` to both. Rotations matched but rescale still differed.
- Long code packing format mismatch: Verified GPU format is self-consistent.
- Shared memory layout corruption: Verified struct sizes and offsets match between host and device.
- CUDA printf format bugs: `%zu` doesn't work on Windows CUDA, causing garbled output that initially appeared as wrong cluster IDs.

**The breakthrough:** Seeding random generators made rotation matrices identical between CPU and GPU. With identical rotations, `v_mx` and `delta` matched exactly. But `ip_o_oa` still differed, meaning the codes were the same but the factor computation was different.

**Root cause:** Reading the CPU encoder source (`caq_encoder.h` line 231):
```cpp
caq.rescale_vmx_to1();
```
which executes (line 35):
```cpp
fac_rescale *= v_mx;
```

The CPU encoder multiplies `fac_rescale` by `v_mx` as part of the `rescale_vmx_to1` normalization step. The GPU fused encoder computed `fac_rescale = o_l2sqr / ip_o_oa` but never applied this multiplication.

**The fix:** One line change in both kernel variants:
```cuda
// Before:
float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa) : 0.0f;

// After:
float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa * v_mx) : 0.0f;
```

**Impact:** Recall jumped from 41% to 89.95%, matching CPU's 89.91%.

### 4.5 Why This Bug Was Hard to Find

1. **The formula looked correct in isolation.** `rescale = o_l2sqr / ip_o_oa` is the textbook formula. The `* v_mx` step is a non-obvious normalization applied in a separate function (`rescale_vmx_to1`) that also modifies `delta`, `ip_o_oa`, `oa_l2sqr`, `v_mi`, and `v_mx` together.

2. **The GPU's rescale values were "reasonable."** `rescale ≈ 1.0` is a valid value (it means the quantized approximation closely matches the original). Only by comparing against the CPU's `rescale ≈ 0.15` for the SAME vector with the SAME rotation was the discrepancy visible.

3. **Multiple confounding factors.** Different random rotations between CPU and GPU builds produced different quantization quality, making it hard to isolate whether the issue was in the formula or the input data. Seeding the random generator to force identical rotations was the key step that enabled the comparison.

4. **Distance overestimation is normal.** Even the CPU's `compAccurateDist` overestimates by 4x (2.35 vs 0.57 brute-force). The SAQ distance is an approximation — what matters is that the overestimation is CONSISTENT across vectors (preserving ranking), which the `v_mx` multiplication ensures.

## 5. Performance Analysis

### 5.1 Encode Pipeline

| Phase | Original | Optimized |
|-------|----------|-----------|
| GPU alloc | 900ms | 2ms |
| GPU scatter | 340ms | 1.5ms |
| GPU kernels | 118ms | 154ms (fused, includes cublasSgeam) |
| H2D upload | 120ms | 120ms |
| CPU prep | 77ms | 77ms |
| **Total** | **1,555ms** | **355ms** |
| CPU 8T | 500-640ms | — |

The fused encode kernel is slightly slower than the original separate kernels (154ms vs 118ms) because of the `cublasSgeam` segment extraction overhead. But the elimination of intermediate buffers (`d_residuals`, `d_codes`) reduces memory allocations and global memory traffic.

### 5.2 Search Pipeline

| Component | Time (Q=1000) | Notes |
|-----------|--------------|-------|
| Centroid search (CPU) | 217ms | Single-threaded FlatInitializer |
| Query rotation (CPU) | 28ms | Eigen matrix multiply per segment |
| Upload | 3ms | Queries + descriptors + constants |
| Search kernel | 75ms | 200K blocks, all-vectors accurate distance |
| Merge kernel | 21ms | Per-query selection sort |
| Download | <1ms | 100K uint32 results |
| **Total** | **368ms** | **2,716 QPS** |

The search kernel is the dominant GPU cost (75ms). The current all-vectors approach computes accurate distance for every vector in every searched cluster. Adding stage 2 fast-distance filtering would reduce this to ~10-20ms by evaluating only ~20% of vectors accurately.

### 5.3 Bottleneck Analysis

The centroid search (217ms, CPU single-threaded) is the largest single cost. Parallelizing with OpenMP or moving to GPU would reduce total search time to ~150ms (~6,700 QPS).

## 6. Commit History

```
e8ecc9b feat(gpu): fix rescale_vmx_to1 in fused encode — GPU recall matches CPU
aa70697 debug(gpu): add CPU-side compAccurateDist reference comparison
eb3b02d debug(gpu): add distance diagnostic comparing CPU vs GPU search
93b310c feat(gpu): fix codebook bit order and add stage 3 accurate distance
41071d2 feat(gpu): implement stage 3 accurate distance in search kernel
b843332 feat(gpu): add GPU batch search with benchmark
bafb690 feat(gpu): add search_batch host orchestration
4a7598e feat(gpu): implement GPU batch search kernel with LUT fastscan
aae2278 feat(gpu): add search descriptor types and kernel declarations
977f891 chore(gpu): delete gpu_packer (folded into fused encode + scatter)
6e83d7e feat(gpu): wire fused encode into construct, remove intermediate buffers
3f3fa1d feat(gpu): implement fused CAQ encode kernel (subtract + encode + pack)
7164236 feat(gpu): replace per-cluster scatter loop with GPU scatter kernels
af00456 feat(gpu): add scatter kernels for pool-based data layout
67260dd feat(gpu): integrate GpuMemoryPool into construct pipeline
5c78bba refactor(gpu): strip ownership from GpuSaqCluData (pool will own memory)
251e3c0 feat(gpu): add GpuMemoryPool for bulk device memory allocation
```

## 7. Files Modified/Created

### New Files
| File | Purpose |
|------|---------|
| `include/saq/gpu/gpu_memory_pool.h` | Bulk device memory allocation with offset tables |
| `include/saq/gpu/gpu_scatter.cuh` | Scatter kernel declarations |
| `src/gpu/gpu_scatter.cu` | Scatter + fastscan reorder kernel implementations |
| `include/saq/gpu/gpu_searcher.cuh` | Search descriptor types and kernel declarations |
| `src/gpu/gpu_search.cu` | Search + merge kernel implementations |
| `src/gpu/gpu_ivf_search.cpp` | Host-side search orchestration |

### Modified Files
| File | Change |
|------|--------|
| `include/saq/gpu/gpu_cluster_data.cuh` | Stripped to lightweight view (pool owns memory) |
| `include/saq/gpu/gpu_ivf.h` | Added pool member, search_batch method |
| `src/gpu/gpu_ivf_construct.cpp` | Pool allocation, scatter kernels, fused encode |
| `src/gpu/gpu_encoder.cu` | Fused encode with L1+L2 fusion, rescale_vmx_to1 fix |
| `include/saq/gpu/gpu_encoder.cuh` | Updated signatures for fused encode |
| `samples/gpu_benchmark_sample.cpp` | Search benchmark and diagnostic code |
| `src/CMakeLists.txt` | Added new source files |

### Deleted Files
| File | Reason |
|------|--------|
| `src/gpu/gpu_packer.cu` | Pack kernels folded into fused encode |
| `include/saq/gpu/gpu_packer.cuh` | Declarations moved to encoder/scatter |

## 8. Future Optimizations

1. **Stage 2 fast-distance filtering:** Re-enable the LUT-based fast screening to reduce the number of accurate distance computations from ~24 to ~5 per cluster. Requires calibrating the fast distance formula constants for the GPU's float LUT (vs CPU's int16 LUT).

2. **Parallel centroid search:** The 217ms CPU centroid search is single-threaded. OpenMP parallelization or GPU-based centroid search would cut total search time nearly in half.

3. **Batch query rotation on GPU:** The 28ms CPU query rotation could be done as a GPU GEMM batch for large Q.

4. **Warp-parallel merge:** The merge kernel uses a single thread per query. A warp-parallel partial sort would be faster for large candidate counts.

5. **Persistent search kernel:** Process multiple (query, cluster) pairs per block to improve GPU occupancy for small Q values.
