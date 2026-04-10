# SAQ Python Packaging + Test Bench Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Revision History

- **2026-04-10 (v2)** — Tasks 6-8 revised after in-progress attempt hit bugs:
  - `FloatVec` is a row vector (`Matrix<float, 1, Dynamic>`), not a column
    vector. All `.transpose()` calls in broadcast subtraction removed;
    rotator multiplication order flipped to `row_vec * P` / `row_vec * P^T`.
  - `decompress()` switched to **cached raw codes (Option B)** instead of
    inverting the SIMD-scrambled bit-packed storage. A new `raw_codes_`
    member on `IVF` stores per-cluster, per-segment `uint16_t` code matrices
    during `fit()` (via a threaded output buffer through `construct_impl` →
    `quantize_cluster` → `quantize`). Memory cost ~N×D×2 bytes, but only
    paid on `fit()` — `construct()` leaves `raw_codes_` empty.
  - Task 5 CMake fix: GCC needs `-mfma` alongside `-mavx512f` for Eigen
    (was a pre-existing Linux build issue).
  - Faiss on Linux comes from conda, not system packages: use
    `-Dfaiss_DIR=$CONDA_PREFIX/share/faiss -DCMAKE_PREFIX_PATH=$CONDA_PREFIX`.
- **2026-04-10 (v1)** — Initial plan.

## Already Completed (Phase 1, partial — on `feat/ivf-fit-api` branch)

Tasks 1-5 are complete and committed in the `feat/ivf-fit-api` worktree at
`/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/saq-ivf-fit-api`:

- ✅ Task 1: Preprocessing headers (`pca.h`, `kmeans.h`, `preprocessing.h`)
- ✅ Task 2: Eigen preprocessing implementations (SVD-based PCA, Lloyd's k-means)
- ✅ Task 3: Faiss preprocessing implementations
- ✅ Task 4: `preprocessing.cpp` orchestrator
- ✅ Task 5: CMake `SAQ_USE_FAISS` option + `-mfma` flag fix

The `saq` static library builds cleanly on Linux with conda Faiss 1.14.1.
Task 6 onwards should resume from this state.

---

**Goal:** Add `IVF::fit()` + `IVF::decompress()` + a self-contained preprocessing module to the SAQ C++ library, expose them via identical pybind11 bindings on all four branches, then unify the vector-quantization test bench around a `BaseSearchIndex` ABC that benchmarks all VQ methods on recall@k, QPS, memory, and reconstruction MSE.

**Architecture:** The SAQ library gains a `preprocessing/` module (Faiss on Linux, Eigen fallback on Windows) compiled into the `saq` static library behind a `SAQ_USE_FAISS` CMake flag; `IVF::fit()` chains `fit_ivf_preprocessing()` → `construct()` in one call. The test bench adds a `BaseSearchIndex` ABC and wrapper classes (`SaqIndex`, `FlatQuantizedIndex`, `IvfQuantizedIndex`, `FaissIvfPqIndex`) so every VQ method is benchmarked on the same axes. All four SAQ wheels (cpu/gpu/codebook/gpu-codebook) expose the same Python API surface, selected by which wheel is installed.

**Tech Stack:** C++20, Eigen 3.4.0, Faiss (Linux), pybind11 v2.12.0, Python 3.10+, numpy, faiss-cpu (Python), pytest, pandas, matplotlib.

---

## File Structure

### SAQ C++ library (branch: `feat/ivf-fit-api` from `main`)

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `include/saq/preprocessing/pca.h` | `PCAResult` struct, `PCAFit` class declaration |
| Create | `include/saq/preprocessing/kmeans.h` | `KMeansResult` struct, `KMeans` class declaration |
| Create | `include/saq/preprocessing/preprocessing.h` | `PreprocessingResult` struct, `fit_ivf_preprocessing()` umbrella |
| Create | `src/preprocessing/pca_eigen.cpp` | Eigen BDCSVD PCA implementation (Windows fallback) |
| Create | `src/preprocessing/pca_faiss.cpp` | Faiss PCAMatrix implementation (Linux default) |
| Create | `src/preprocessing/kmeans_eigen.cpp` | Lloyd's + k-means++ Eigen implementation (Windows fallback) |
| Create | `src/preprocessing/kmeans_faiss.cpp` | Faiss Kmeans implementation (Linux default) |
| Create | `src/preprocessing/preprocessing.cpp` | `fit_ivf_preprocessing()` orchestrator |
| Modify | `include/index/ivf_index.h` | Add `fit()`, `decompress()`, `find_vector()` declarations + `id_to_location_` map member |
| Modify | `src/ivf_index.cpp` | Implement `fit()`, `decompress()`, `find_vector()`, populate `id_to_location_` in `construct()` |
| Modify | `CMakeLists.txt` | Add `SAQ_USE_FAISS` option, Faiss find_package, conditional source registration |
| Modify | `src/CMakeLists.txt` | Register preprocessing sources conditionally |
| Modify | `python/bindings/saq_bindings.cpp` | Add `.def("fit", ...)`, `.def("decompress", ...)`, `.def("set_codebooks", ...)`, `.def("set_gaussian_codebooks", ...)` |
| Create | `tests/test_fit_decompress.cpp` | C++ unit test for `IVF::fit()` and `IVF::decompress()` |

### Branch propagation

| Action | Branch | Steps |
|--------|--------|-------|
| Merge | `feat/ivf-fit-api` → `main` | After all tests pass |
| Merge | `main` → `gpu` | Carry preprocessing + bindings |
| Merge | `main` → `feat/optimal-codebook` | Carry preprocessing + bindings |
| Create | `gpu-codebook` from `gpu` | Cherry-pick codebook binding stubs |

### vector-quantization test bench (branch: `feat/test-bench-search-index`)

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/haag_vq/methods/base_search_index.py` | `BaseSearchIndex` ABC — primary benchmark interface |
| Create | `src/haag_vq/methods/search/__init__.py` | Package init, re-exports all wrappers |
| Create | `src/haag_vq/methods/search/saq_index.py` | `SaqIndex` wrapping `saq.IVF` / `saq.GpuIVF` |
| Create | `src/haag_vq/methods/search/flat_quantized_index.py` | `FlatQuantizedIndex` — brute-force wrapper over any `BaseQuantizer` |
| Create | `src/haag_vq/methods/search/ivf_quantized_index.py` | `IvfQuantizedIndex` — k-means IVF shell around any `BaseQuantizer` |
| Create | `src/haag_vq/methods/search/faiss_ivfpq_index.py` | `FaissIvfPqIndex` — Faiss IndexIVFPQ baseline |
| Create | `src/haag_vq/benchmarks/search_bench.py` | `benchmark_index()`, `sweep_bpd()`, `compare_methods()`, `pareto_plot()` |
| Modify | `src/haag_vq/benchmarks/run_benchmarks.py` | Rewrite to dispatch all methods through `BaseSearchIndex` |
| Delete | `src/haag_vq/methods/saq.py` | Remove pure-Python placeholder |
| Modify | `src/haag_vq/methods/__init__.py` | Remove `saq.py` import; add search subpackage re-exports |
| Modify | `tests/test_saq.py` | Rewrite against `SaqIndex` / `BaseSearchIndex` contract |
| Create | `tests/test_flat_quantized.py` | Contract tests for `FlatQuantizedIndex` |
| Create | `tests/test_ivf_quantized.py` | Contract tests for `IvfQuantizedIndex` |

---

## Phase 1 — SAQ C++ Library: Preprocessing Module

### Task 1: Preprocessing headers

**Files:**
- Create: `include/saq/preprocessing/pca.h`
- Create: `include/saq/preprocessing/kmeans.h`
- Create: `include/saq/preprocessing/preprocessing.h`

- [ ] **Step 1: Create `pca.h`**

```cpp
#pragma once
// include/saq/preprocessing/pca.h

#include "saq/defines.h"

namespace saq {

struct PCAResult {
    FloatVec   mean;      // (D,)  — training mean
    FloatRowMat rotation; // (D,D) — orthogonal rotation matrix (columns = eigenvectors)
    FloatVec   variances; // (D,)  — per-dim variance in rotated space
};

/// Compute PCA from a data matrix.  Implementation lives in pca_eigen.cpp
/// or pca_faiss.cpp depending on SAQ_USE_FAISS compile flag.
class PCAFit {
public:
    /// Fit PCA to training data X (N, D). Returns rotation + mean + variances.
    PCAResult fit(const FloatRowMat& X) const;
};

} // namespace saq
```

- [ ] **Step 2: Create `kmeans.h`**

```cpp
#pragma once
// include/saq/preprocessing/kmeans.h

#include <cstdint>
#include "saq/defines.h"

namespace saq {

struct KMeansResult {
    FloatRowMat          centroids;   // (K, D)
    std::vector<PID>     assignments; // (N,) — cluster index per training vector
};

/// Run K-means clustering.  Implementation lives in kmeans_eigen.cpp or
/// kmeans_faiss.cpp depending on SAQ_USE_FAISS compile flag.
class KMeans {
public:
    explicit KMeans(int K, int max_iter = 25, int seed = 0);
    KMeansResult fit(const FloatRowMat& X) const;

private:
    int K_, max_iter_, seed_;
};

} // namespace saq
```

- [ ] **Step 3: Create `preprocessing.h`**

```cpp
#pragma once
// include/saq/preprocessing/preprocessing.h

#include "saq/preprocessing/pca.h"
#include "saq/preprocessing/kmeans.h"

namespace saq {

struct PreprocessingResult {
    PCAResult    pca;
    KMeansResult kmeans;
};

/// Run full IVF preprocessing: PCA (optional) then K-means on (possibly
/// rotated) data.  If apply_pca=false the PCAResult fields are identity
/// (zero mean, identity rotation, ones variances).
PreprocessingResult fit_ivf_preprocessing(
    const FloatRowMat& X,
    int   K,
    int   seed      = 0,
    bool  apply_pca = true
);

} // namespace saq
```

- [ ] **Step 4: Verify headers compile in isolation (no source yet)**

From WSL bash, in the SAQ root:
```bash
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ
# Compile a minimal .cpp that includes the three headers through the cmake include path
cat > /tmp/pp_header_check.cpp << 'EOF'
#include "saq/preprocessing/preprocessing.h"
int main() {}
EOF
# Use same flags as the library build to verify includes resolve
g++ -std=c++20 -I include -I build/_deps/eigen-src /tmp/pp_header_check.cpp -fsyntax-only
```
Expected: exits 0, no errors.

---

### Task 2: Eigen fallback implementations (Windows-compatible, always compilable)

**Files:**
- Create: `src/preprocessing/pca_eigen.cpp`
- Create: `src/preprocessing/kmeans_eigen.cpp`

- [ ] **Step 1: Create `pca_eigen.cpp`**

```cpp
// src/preprocessing/pca_eigen.cpp
// Compiled only when SAQ_USE_FAISS=OFF (Windows default).
// Uses Eigen::BDCSVD for PCA. Cap: ~200K vectors recommended.
#ifndef SAQ_USE_FAISS

#include "saq/preprocessing/pca.h"
#include <Eigen/SVD>

namespace saq {

PCAResult PCAFit::fit(const FloatRowMat& X) const {
    PCAResult result;
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());

    // 1. Compute mean
    result.mean = X.colwise().mean();

    // 2. Center data
    FloatRowMat Xc = X.rowwise() - result.mean.transpose();

    // 3. Covariance via SVD of centered data (economy mode)
    //    SVD(Xc) = U S Vt  =>  cov = Vt^T diag(S^2/(N-1)) Vt
    Eigen::BDCSVD<FloatRowMat> svd(Xc, Eigen::ComputeThinV);
    result.rotation = svd.matrixV(); // (D, D) — columns = eigenvectors
    FloatVec sv     = svd.singularValues(); // (min(N,D),)
    int K = static_cast<int>(sv.size());
    result.variances.resize(D);
    for (int i = 0; i < K;  ++i) result.variances[i] = sv[i] * sv[i] / (N - 1);
    for (int i = K; i < D; ++i) result.variances[i] = 0.0f;

    return result;
}

} // namespace saq
#endif // !SAQ_USE_FAISS
```

- [ ] **Step 2: Create `kmeans_eigen.cpp`**

```cpp
// src/preprocessing/kmeans_eigen.cpp
// Compiled only when SAQ_USE_FAISS=OFF (Windows default).
// Lloyd's algorithm with k-means++ initialization.
#ifndef SAQ_USE_FAISS

#include "saq/preprocessing/kmeans.h"
#include <random>
#include <limits>

namespace saq {

KMeans::KMeans(int K, int max_iter, int seed)
    : K_(K), max_iter_(max_iter), seed_(seed) {}

KMeansResult KMeans::fit(const FloatRowMat& X) const {
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());
    std::mt19937 rng(static_cast<uint32_t>(seed_));

    // --- k-means++ initialization ---
    FloatRowMat centroids(K_, D);
    std::uniform_int_distribution<int> uni(0, N - 1);
    centroids.row(0) = X.row(uni(rng));

    for (int c = 1; c < K_; ++c) {
        // Squared distances from each point to its nearest centroid so far
        FloatVec d2(N);
        for (int i = 0; i < N; ++i) {
            float best = std::numeric_limits<float>::max();
            for (int j = 0; j < c; ++j) {
                float dist = (X.row(i) - centroids.row(j)).squaredNorm();
                if (dist < best) best = dist;
            }
            d2[i] = best;
        }
        std::discrete_distribution<int> weighted(d2.data(), d2.data() + N);
        centroids.row(c) = X.row(weighted(rng));
    }

    // --- Lloyd's iterations ---
    std::vector<PID> assignments(N, 0);
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Assignment step
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            float best_d = std::numeric_limits<float>::max();
            PID   best_c = 0;
            for (int c = 0; c < K_; ++c) {
                float d = (X.row(i) - centroids.row(c)).squaredNorm();
                if (d < best_d) { best_d = d; best_c = static_cast<PID>(c); }
            }
            if (best_c != assignments[i]) { assignments[i] = best_c; changed = true; }
        }
        if (!changed) break;

        // Update step
        FloatRowMat new_centroids = FloatRowMat::Zero(K_, D);
        std::vector<int> counts(K_, 0);
        for (int i = 0; i < N; ++i) {
            new_centroids.row(assignments[i]) += X.row(i);
            ++counts[assignments[i]];
        }
        for (int c = 0; c < K_; ++c) {
            if (counts[c] > 0)
                centroids.row(c) = new_centroids.row(c) / static_cast<float>(counts[c]);
            // else: keep old centroid (degenerate cluster)
        }
    }

    return KMeansResult{std::move(centroids), std::move(assignments)};
}

} // namespace saq
#endif // !SAQ_USE_FAISS
```

---

### Task 3: Faiss implementations (Linux default)

**Files:**
- Create: `src/preprocessing/pca_faiss.cpp`
- Create: `src/preprocessing/kmeans_faiss.cpp`

- [ ] **Step 1: Create `pca_faiss.cpp`**

```cpp
// src/preprocessing/pca_faiss.cpp
// Compiled only when SAQ_USE_FAISS=ON (Linux default).
#ifdef SAQ_USE_FAISS

#include "saq/preprocessing/pca.h"
#include <faiss/VectorTransform.h>
#include <faiss/utils/distances.h>
#include <cstring>

namespace saq {

PCAResult PCAFit::fit(const FloatRowMat& X) const {
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());

    faiss::PCAMatrix pca(D, D, /*eigenvalue power*/ 0, /*random rotation*/ false);
    // Faiss expects row-major float* (which Eigen FloatRowMat is)
    pca.train(N, X.data());

    PCAResult result;

    // Mean
    result.mean.resize(D);
    std::memcpy(result.mean.data(), pca.mean.data(), D * sizeof(float));

    // Rotation matrix: Faiss stores A (D x D) in row-major such that
    // x_rotated = (x - mean) @ A^T  (Faiss convention).
    // We want rotation R s.t. x_rotated = (x - mean) @ R.
    // So R = A^T, and our R is (D,D) with columns = eigenvectors.
    result.rotation.resize(D, D);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            result.rotation(j, i) = pca.A[static_cast<size_t>(i) * D + j];

    // Variances: Faiss stores eigenvalues in pca.eigenvalues (D,) descending
    result.variances.resize(D);
    std::memcpy(result.variances.data(), pca.eigenvalues.data(), D * sizeof(float));

    return result;
}

} // namespace saq
#endif // SAQ_USE_FAISS
```

- [ ] **Step 2: Create `kmeans_faiss.cpp`**

```cpp
// src/preprocessing/kmeans_faiss.cpp
// Compiled only when SAQ_USE_FAISS=ON (Linux default).
#ifdef SAQ_USE_FAISS

#include "saq/preprocessing/kmeans.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <cstring>

namespace saq {

KMeans::KMeans(int K, int max_iter, int seed)
    : K_(K), max_iter_(max_iter), seed_(seed) {}

KMeansResult KMeans::fit(const FloatRowMat& X) const {
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());

    faiss::ClusteringParameters cp;
    cp.niter    = max_iter_;
    cp.seed     = seed_;
    cp.verbose  = false;

    faiss::Clustering kmeans(D, K_, cp);
    faiss::IndexFlatL2 index(D);
    kmeans.train(N, X.data(), index);

    KMeansResult result;

    // Centroids (K, D)
    result.centroids.resize(K_, D);
    std::memcpy(result.centroids.data(), kmeans.centroids.data(),
                static_cast<size_t>(K_) * D * sizeof(float));

    // Assignments: search each training point against centroids
    result.assignments.resize(N);
    std::vector<faiss::idx_t> ids(N);
    std::vector<float> dists(N);
    index.search(N, X.data(), 1, dists.data(), ids.data());
    for (int i = 0; i < N; ++i)
        result.assignments[i] = static_cast<PID>(ids[i]);

    return result;
}

} // namespace saq
#endif // SAQ_USE_FAISS
```

---

### Task 4: `preprocessing.cpp` orchestrator

**Files:**
- Create: `src/preprocessing/preprocessing.cpp`

- [ ] **Step 1: Create `preprocessing.cpp`**

```cpp
// src/preprocessing/preprocessing.cpp
#include "saq/preprocessing/preprocessing.h"

namespace saq {

PreprocessingResult fit_ivf_preprocessing(
    const FloatRowMat& X,
    int   K,
    int   seed,
    bool  apply_pca)
{
    PreprocessingResult result;

    if (apply_pca) {
        PCAFit pca_fit;
        result.pca = pca_fit.fit(X);
        // Apply PCA rotation to training data before clustering
        FloatRowMat X_pca = (X.rowwise() - result.pca.mean.transpose())
                            * result.pca.rotation;
        KMeans km(K, /*max_iter=*/25, seed);
        result.kmeans = km.fit(X_pca);
    } else {
        // Identity PCA result
        const int D = static_cast<int>(X.cols());
        result.pca.mean      = FloatVec::Zero(D);
        result.pca.rotation  = FloatRowMat::Identity(D, D);
        result.pca.variances = FloatVec::Ones(D);

        KMeans km(K, /*max_iter=*/25, seed);
        result.kmeans = km.fit(X);
    }

    return result;
}

} // namespace saq
```

---

### Task 5: CMake changes — `SAQ_USE_FAISS` option and source registration

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Add `SAQ_USE_FAISS` block and fix GCC AVX-512 flags in `CMakeLists.txt`**

The `main` branch does not have a `SAQ_BUILD_CUDA` block — insert the Faiss
block after the OpenMP block, before `if(SAQ_BUILD_TESTS)`.

Also: the existing `if(SAQ_REQUIRE_AVX512)` branch for non-MSVC compilers
passes `-mavx512f -mavx512bw -mavx512dq -mavx512vl` but Eigen 3.4.0 requires
`-mfma` alongside `-mavx512f`. Add `-mfma` to the GCC/Clang flag list.

Change:
```cmake
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(saq PRIVATE /arch:AVX512)
  else()
    target_compile_options(saq PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl)
  endif()
  target_compile_definitions(saq PUBLIC SAQ_REQUIRE_AVX512=1)
endif()
```

to:
```cmake
if(SAQ_REQUIRE_AVX512)
  if(MSVC)
    target_compile_options(saq PRIVATE /arch:AVX512)
  else()
    # GCC/Clang need -mfma alongside -mavx512f (Eigen requirement)
    target_compile_options(saq PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
  endif()
  target_compile_definitions(saq PUBLIC SAQ_REQUIRE_AVX512=1)
endif()
```

Then insert the Faiss block after the OpenMP block:
```cmake
# ---- Faiss / Eigen preprocessing dispatch ----
if(WIN32)
    set(SAQ_USE_FAISS_DEFAULT OFF)
else()
    set(SAQ_USE_FAISS_DEFAULT ON)
endif()
option(SAQ_USE_FAISS "Use Faiss for PCA and k-means (Linux default ON, Windows default OFF)"
       ${SAQ_USE_FAISS_DEFAULT})

if(SAQ_USE_FAISS)
    find_package(faiss REQUIRED)
    target_link_libraries(saq PRIVATE faiss)
    target_compile_definitions(saq PRIVATE SAQ_USE_FAISS)
    message(STATUS "SAQ_USE_FAISS=ON: using Faiss for PCA and k-means")
else()
    message(STATUS "SAQ_USE_FAISS=OFF: using Eigen BDCSVD fallback for PCA and k-means")
endif()
```

On Linux with conda-installed Faiss, the CMake invocation needs to point at
the conda environment's CMake config:

```bash
cmake -B build -DSAQ_USE_FAISS=ON \
  -Dfaiss_DIR=$CONDA_PREFIX/share/faiss \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
```

- [ ] **Step 2: Add preprocessing sources to `src/CMakeLists.txt`**

Append after the existing `target_sources(saq PRIVATE ...)` block:

```cmake
# Preprocessing module (shared across all variants)
target_sources(saq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/preprocessing.cpp
)

if(SAQ_USE_FAISS)
    target_sources(saq PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/pca_faiss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/kmeans_faiss.cpp
    )
else()
    target_sources(saq PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/pca_eigen.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/kmeans_eigen.cpp
    )
endif()
```

- [ ] **Step 3: Verify library builds with Eigen fallback (Windows)**

```bash
cmd.exe /c "cmake -B build -G Ninja -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON -DSAQ_USE_FAISS=OFF -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl && cmake --build build --target saq 2>&1"
```
Expected: build succeeds, `saq.lib` produced, no errors about missing Faiss.

---

### Task 6: `IVF::fit()` and `IVF::decompress()` — header additions

**Design note (revised 2026-04-10):** `decompress()` uses **cached raw codes**
(`Option B`) rather than inverting the bit-packed storage. The bit-packed
storage uses SIMD-scrambled layouts (8-byte swap without fastscan,
`fastscan::pack_codes` with fastscan) that are tedious to invert and
error-prone. Caching the raw `Eigen::VectorXi` codes per segment per vector
during `fit()` costs ~N×D×2 bytes (uint16_t, supports up to 13 bits/dim)
but makes reconstruction trivially correct. The cache is **only populated by
`fit()`**, not `construct()`, so enterprise users who bring preprocessed data
pay zero memory overhead.

Another design note: `FloatVec` in this codebase is
`Eigen::Matrix<float, 1, Dynamic>` — a **row vector**, not a column vector.
Broadcast subtraction uses `X.rowwise() - row_vec` (no `.transpose()`).
Linux GCC AVX-512 builds require `-mfma` alongside `-mavx512f` (Eigen
requirement — fixed in the `CMakeLists.txt` modification in Task 5).

**Files:**
- Modify: `include/index/ivf_index.h`

- [ ] **Step 1: Add `id_to_location_`, `raw_codes_` members and new method declarations to `IVF`**

In the `protected:` section of `IVF`, after `std::unique_ptr<SaqDataMaker> saq_data_maker_;`, add:

```cpp
// Maps global vector ID → {cluster_idx, local_idx_within_cluster}.
// Populated at the end of construct(). Used by decompress() to locate
// a vector's slot in parallel_clusters_.
std::unordered_map<PID, std::pair<size_t, size_t>> id_to_location_;

// PCA state stored during fit() for use in decompress() inverse rotation.
bool          pca_applied_  = false;
FloatVec      pca_mean_;       // (1, D) original-space mean (row vector)
FloatRowMat   pca_rotation_;   // (D, D) rotation applied during fit

// Raw quantization codes cached during fit() for decompress().
// Layout: raw_codes_[cluster_idx][segment_idx] is a (num_vectors_in_cluster,
// num_dim_padded_for_segment) uint16_t matrix where row k holds the integer
// codes for the k-th vector in that cluster for that segment.
// Only populated when fit() was called; empty when construct() was called
// directly (enterprise path — no decompress support).
using RawCodeMat = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
std::vector<std::vector<RawCodeMat>> raw_codes_;
```

In the `public:` section, after `set_variance()`:

```cpp
/// Self-contained preprocessing + construction from raw (N, D) vectors.
/// Handles PCA, k-means, and construct() internally. Also caches raw
/// quantization codes so decompress() can reconstruct vectors later.
void fit(
    const FloatRowMat& X,
    bool apply_pca    = true,
    int  K            = 4096,
    int  seed         = 0,
    int  num_threads  = 8
);

/// Reconstruct approximate vectors from cached raw codes.
/// ids: global vector IDs (as returned by search()).
/// Returns (ids.size(), num_dim_) matrix in original (pre-PCA) space.
/// REQUIRES fit() to have been called (not construct() alone). Throws if
/// raw_codes_ is empty.
FloatRowMat decompress(const std::vector<PID>& ids) const;
```

Also add these includes to the top of `ivf_index.h`:
```cpp
#include <unordered_map>
#include "saq/preprocessing/preprocessing.h"
```

- [ ] **Step 2: Verify header compiles**

```bash
export PATH=/home/rohil/miniconda3/bin:$PATH
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/saq-ivf-fit-api
cmake --build build --target saq 2>&1 | tail -10
```
Expected: build succeeds (unimplemented methods not called yet).

---

### Task 7: `IVF::fit()`, `IVF::decompress()`, and raw-code caching — source implementations

**Design:** decompress() reads from cached raw codes (Option B, described in
Task 6 notes). Caching is threaded through the encode path via an optional
output buffer on `QuantizerCluster::quantize` and
`SAQuantizer::quantize_cluster`. The cache is populated only during
`IVF::fit()`, not `construct()`, so enterprise users pay zero overhead.

**Files to modify:**
- `include/saq/quantizer.h` — add `raw_codes_out` parameter to `quantize()`
- `include/saq/saq_quantizer.h` — add `raw_codes_out` parameter to `quantize_cluster()`
- `src/ivf_index.cpp` — implement `fit()`, `decompress()`, populate `id_to_location_`

---

- [ ] **Step 1: Add optional `raw_codes_out` parameter to `QuantizerCluster::quantize()`**

In `include/saq/quantizer.h`, change the `virtual void quantize(...)` signature
on `QuantizerCluster` to:

```cpp
virtual void quantize(const FloatRowMat &or_vecs, const FloatVec &centroid,
                      CAQClusterData &clus,
                      Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* raw_codes_out = nullptr) const {
    CHECK_EQ(or_vecs.cols(), static_cast<Eigen::Index>(num_dim_pad_))
        << "Input vector dimension does not match quantizer dimension";
    CHECK_EQ(centroid.cols(), static_cast<Eigen::Index>(num_dim_pad_))
        << "Centroid dimension does not match quantizer dimension";

    FloatRowMat o_vecs;
    if (data_->rotator) {
        clus.centroid() = centroid * data_->rotator->get_P();
        o_vecs = (or_vecs * data_->rotator->get_P()).rowwise() - clus.centroid();
    } else {
        clus.centroid() = centroid;
        o_vecs = or_vecs.rowwise() - clus.centroid();
    }

    const size_t num_points = clus.num_vec();
    CHECK(data_->cfg.quant_type == BaseQuantType::CAQ) << "Only CAQ is supported for DataQuantizer";
    CAQEncoder encoder(num_dim_pad_, num_bits_, data_->cfg);
    ClusterPacker packer(num_dim_pad_, num_bits_, clus, data_->cfg.use_fastscan);

    if (raw_codes_out) {
        raw_codes_out->resize(static_cast<Eigen::Index>(num_points),
                              static_cast<Eigen::Index>(num_dim_pad_));
    }

    QuantBaseCode base_code;
    for (size_t i = 0; i < num_points; ++i) {
        const auto &curr_vec = o_vecs.row(i);
        encoder.encode_and_fac(curr_vec, base_code, &centroid);
        // Cache raw integer code BEFORE store_and_pack (which may move it).
        if (raw_codes_out) {
            for (Eigen::Index d = 0; d < static_cast<Eigen::Index>(num_dim_pad_); ++d) {
                (*raw_codes_out)(static_cast<Eigen::Index>(i), d) =
                    static_cast<uint16_t>(base_code.code[d]);
            }
        }
        packer.store_and_pack(i, base_code);
        metrics_.norm_ip_o_oa.insert(base_code.norm_ip_o_oa);
    }
    packer.finalize_and_store();
}
```

- [ ] **Step 2: Thread `raw_codes_out` through `SAQuantizer::quantize_cluster()`**

In `include/saq/saq_quantizer.h`, change `quantize_cluster()` to accept a
per-segment output buffer vector:

```cpp
void quantize_cluster(const FloatRowMat &data, const FloatVec &centroid, const std::vector<PID> &IDs,
                      SaqCluData &saq_clus,
                      std::vector<Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>* raw_codes_per_segment = nullptr) {
    CHECK_EQ(saq_clus.num_segments_, data_quans_.size());
    std::copy(IDs.begin(), IDs.end(), saq_clus.ids());

    if (raw_codes_per_segment) {
        raw_codes_per_segment->resize(saq_clus.num_segments_);
    }

    const size_t num_points = saq_clus.num_vec_;
    for (size_t ci = 0, offset = 0; ci < saq_clus.num_segments_; ++ci) {
        auto &clus = saq_clus.get_segment(ci);
        const size_t copy_size = std::min(clus.num_dim_padded_, num_dim_ - offset);

        FloatRowMat vecs(static_cast<Eigen::Index>(num_points),
                         static_cast<Eigen::Index>(clus.num_dim_padded_));
        vecs.setZero();
        for (size_t r = 0; r < num_points; ++r) {
            auto id = clus.ids()[r];
            vecs.row(static_cast<Eigen::Index>(r)).head(static_cast<Eigen::Index>(copy_size)) =
                data.row(static_cast<Eigen::Index>(id))
                    .segment(static_cast<Eigen::Index>(offset),
                             static_cast<Eigen::Index>(copy_size));
        }

        FloatVec cen(static_cast<Eigen::Index>(clus.num_dim_padded_));
        cen.setZero();
        cen.head(static_cast<Eigen::Index>(copy_size)) =
            centroid.segment(static_cast<Eigen::Index>(offset),
                             static_cast<Eigen::Index>(copy_size));

        auto* seg_out = raw_codes_per_segment ? &(*raw_codes_per_segment)[ci] : nullptr;
        data_quans_[ci]->quantize(vecs, cen, clus, seg_out);
        offset += clus.num_dim_padded_;
    }
}
```

- [ ] **Step 3: Add includes to `src/ivf_index.cpp`**

At the top of `src/ivf_index.cpp`, alongside existing includes:
```cpp
#include <unordered_map>
#include "saq/preprocessing/preprocessing.h"
```

- [ ] **Step 4: Thread `raw_codes_out` through `IVF::construct()`**

The cleanest way to cache raw codes is inside the existing `construct()`
OMP loop so each cluster's codes are captured during its single encoding
pass. We add an optional private helper that accepts the buffer, and
`fit()` becomes a one-liner that passes a pointer while `construct()`
continues to call the helper with `nullptr`.

In `include/index/ivf_index.h`, change the public `construct()` declaration
to forward to an internal helper. Add to the class (`private:` section):

```cpp
private:
    // Shared implementation of construct(). When raw_codes_out is non-null,
    // populates it with per-cluster, per-segment raw uint16_t codes.
    void construct_impl(const FloatRowMat &data, const FloatRowMat &centroids,
                        const PID *cluster_ids, int num_threads, bool use_1_centroid,
                        std::vector<std::vector<RawCodeMat>>* raw_codes_out);
```

And keep the existing public `construct()` as a thin wrapper (`ivf_index.h`
public section, replace the existing declaration):

```cpp
void construct(const FloatRowMat &data, const FloatRowMat &centroids,
               const PID *cluster_ids, int num_threads = 64,
               bool use_1_centroid = false);
```

In `src/ivf_index.cpp`, rename the existing `IVF::construct` implementation
to `IVF::construct_impl` and add the `raw_codes_out` parameter. At the top
of the function, if `raw_codes_out` is non-null, resize it to `num_cen_`
entries. Inside the parallel for loop, pass `&(*raw_codes_out)[i]` to
`saq_quantizer_.quantize_cluster(...)` when `raw_codes_out` is non-null,
otherwise pass `nullptr`. At the end of the function (after the loop and
timing log), add the `id_to_location_` population:

```cpp
// Build id → {cluster_idx, local_idx} map for decompress()
id_to_location_.clear();
id_to_location_.reserve(num_data_);
for (size_t cid = 0; cid < parallel_clusters_.size(); ++cid) {
    const PID* cluster_ids_ptr = parallel_clusters_[cid].ids();
    size_t nv = parallel_clusters_[cid].num_vec_;
    for (size_t local = 0; local < nv; ++local) {
        id_to_location_[cluster_ids_ptr[local]] = {cid, local};
    }
}
```

Add the thin wrapper `IVF::construct` that forwards to `construct_impl`
with `nullptr` for raw_codes_out:

```cpp
void IVF::construct(const FloatRowMat &data, const FloatRowMat &centroids,
                    const PID *cluster_ids, int num_threads, bool use_1_centroid) {
    construct_impl(data, centroids, cluster_ids, num_threads, use_1_centroid, nullptr);
}
```

- [ ] **Step 5: Implement `IVF::fit()` — single-pass encoding with raw-code caching**

Add to `src/ivf_index.cpp`:

```cpp
void IVF::fit(const FloatRowMat& X, bool apply_pca, int K, int seed, int num_threads) {
    num_data_ = static_cast<size_t>(X.rows());
    num_dim_  = static_cast<size_t>(X.cols());
    num_cen_  = static_cast<size_t>(K);
    saq_data_maker_ = std::make_unique<SaqDataMaker>(cfg_, num_dim_);

    // 1. Preprocessing (PCA + k-means on raw data; PCA is applied to X before
    //    k-means inside fit_ivf_preprocessing when apply_pca=true).
    PreprocessingResult pp = fit_ivf_preprocessing(X, K, seed, apply_pca);

    // 2. Store PCA state for decompress() inverse rotation
    pca_applied_  = apply_pca;
    pca_mean_     = pp.pca.mean;        // row vector (1, D)
    pca_rotation_ = pp.pca.rotation;    // (D, D)

    set_variance(pp.pca.variances);

    // 3. Apply PCA to training data + centroids.
    //    FloatVec is a row vector, so no .transpose() in broadcast subtract.
    //    Note: pp.kmeans.centroids was trained on already-rotated data when
    //    apply_pca=true, so centroids are already in rotated space — only X
    //    needs the explicit projection.
    FloatRowMat X_proc = apply_pca
        ? ((X.rowwise() - pp.pca.mean) * pp.pca.rotation).eval()
        : X;
    const FloatRowMat& centroids_proc = pp.kmeans.centroids;

    // 4. Single-pass construction with raw-code caching.
    raw_codes_.clear();  // construct_impl will resize to num_cen_
    construct_impl(X_proc, centroids_proc, pp.kmeans.assignments.data(),
                   num_threads, /*use_1_centroid=*/false, &raw_codes_);
}
```

**Note:** Check the `fit_ivf_preprocessing` implementation in `preprocessing.cpp`
— it runs k-means on **post-PCA** data when `apply_pca=true`, so the returned
centroids are already in rotated space. If this assumption doesn't hold,
`fit()` needs to apply PCA to the centroids as well. Verify before merge.

- [ ] **Step 6: Implement `IVF::decompress()` using cached codes**

Add to `src/ivf_index.cpp`:

```cpp
FloatRowMat IVF::decompress(const std::vector<PID>& ids) const {
    CHECK(saq_data_) << "decompress() called before fit() or construct()";
    CHECK(!raw_codes_.empty())
        << "decompress() requires fit() (not construct()) — raw codes not cached";

    const auto& plan = saq_data_->quant_plan;         // vector<pair<dim_len, bits>>
    const auto& base = saq_data_->base_datas;          // vector<BaseQuantizerData>
    const size_t num_seg = plan.size();

    FloatRowMat result(static_cast<Eigen::Index>(ids.size()),
                       static_cast<Eigen::Index>(num_dim_));
    result.setZero();

    for (size_t i = 0; i < ids.size(); ++i) {
        PID vid = ids[i];
        auto it = id_to_location_.find(vid);
        CHECK(it != id_to_location_.end())
            << "decompress(): ID " << vid << " not found in index";
        auto [cid, local_idx] = it->second;

        const SaqCluData& saq_clu = parallel_clusters_[cid];

        // Build the PCA-rotated reconstruction one segment at a time.
        // FloatVec is a row vector (1, D); use .segment() slicing along cols.
        FloatVec o_rot = FloatVec::Zero(static_cast<Eigen::Index>(num_dim_));
        size_t offset = 0;

        for (size_t seg = 0; seg < num_seg; ++seg) {
            const auto& seg_data  = base[seg];
            const auto& caq_seg   = saq_clu.get_segment(seg);
            const size_t seg_dim  = seg_data.num_dim_pad;
            const size_t num_bits = seg_data.num_bits;

            // Get the effective (un-padded) size for this segment — the part that
            // maps back to real dimensions, not padding.
            const size_t real_size = std::min(seg_dim, num_dim_ - offset);

            if (num_bits == 0) {
                // 0-bit segment: reconstruction is zero (handled by centroid-only below)
                FloatVec seg_vec = FloatVec::Zero(static_cast<Eigen::Index>(seg_dim));
                // Un-rotate segment rotator (if any)
                if (seg_data.rotator) {
                    // FloatVec is row-major, P is (seg_dim, seg_dim).
                    // Inverse: vec * P^T
                    seg_vec = (seg_vec * seg_data.rotator->get_P().transpose()).eval();
                }
                // Add segment centroid (stored in rotated space; un-rotate by the same
                // rotator to get back to PCA-rotated but pre-per-segment-rotator space).
                FloatVec cen_unrot = seg_data.rotator
                    ? (caq_seg.centroid() * seg_data.rotator->get_P().transpose()).eval()
                    : caq_seg.centroid();
                seg_vec += cen_unrot;

                o_rot.segment(static_cast<Eigen::Index>(offset),
                              static_cast<Eigen::Index>(real_size)) =
                    seg_vec.segment(0, static_cast<Eigen::Index>(real_size));
                offset += seg_dim;
                continue;
            }

            // --- num_bits > 0: dequantize from cached raw codes ---

            // Raw codes for this cluster+segment. Row = local_idx, col = dimension.
            const auto& seg_codes = raw_codes_[cid][seg];
            CHECK_EQ(seg_codes.cols(), static_cast<Eigen::Index>(seg_dim));

            // After rescale_vmx_to1() the stored quantizer operates in
            // normalized scale: v_mx=1, v_mi=-1, delta = 2/2^num_bits.
            const float v_mi  = -1.0f;
            const float delta = 2.0f / static_cast<float>(1 << num_bits);

            // get_oa-equivalent reconstruction (row vector):
            //   oa_norm[d] = code[d] * delta + v_mi
            FloatVec oa_norm(static_cast<Eigen::Index>(seg_dim));
            for (size_t d = 0; d < seg_dim; ++d) {
                int code_d = static_cast<int>(
                    seg_codes(static_cast<Eigen::Index>(local_idx),
                              static_cast<Eigen::Index>(d)));
                oa_norm[static_cast<Eigen::Index>(d)] =
                    static_cast<float>(code_d) * delta + v_mi;
            }

            // Scale direction to match original L2 norm (CAQ preserves magnitude).
            // The o_l2norm is stored per vector in the short-factor block.
            size_t block_idx = local_idx / KFastScanSize;
            size_t lane      = local_idx % KFastScanSize;
            float o_l2norm   = caq_seg.factor_o_l2norm(block_idx)[lane];

            float norm_oa = oa_norm.norm();
            if (norm_oa > 1e-9f) {
                oa_norm *= (o_l2norm / norm_oa);
            }

            // Un-rotate per-segment rotator (inverse of `row_vec * P` is `row_vec * P^T`).
            FloatVec oa_unrot = seg_data.rotator
                ? (oa_norm * seg_data.rotator->get_P().transpose()).eval()
                : oa_norm;

            // Add segment centroid (also stored in rotated space).
            FloatVec cen_unrot = seg_data.rotator
                ? (caq_seg.centroid() * seg_data.rotator->get_P().transpose()).eval()
                : caq_seg.centroid();
            oa_unrot += cen_unrot;

            // Write into the PCA-rotated full-dim vector (only the real_size portion).
            o_rot.segment(static_cast<Eigen::Index>(offset),
                          static_cast<Eigen::Index>(real_size)) =
                oa_unrot.segment(0, static_cast<Eigen::Index>(real_size));
            offset += seg_dim;
        }

        // Inverse PCA: x_original = (o_rot * R^T) + mean.
        // o_rot and pca_mean_ are row vectors, pca_rotation_ is (D, D).
        FloatVec x_orig;
        if (pca_applied_) {
            x_orig = (o_rot.head(static_cast<Eigen::Index>(num_dim_))
                      * pca_rotation_.transpose()).eval();
            x_orig += pca_mean_;
        } else {
            x_orig = o_rot.head(static_cast<Eigen::Index>(num_dim_));
        }

        result.row(static_cast<Eigen::Index>(i)) = x_orig;
    }

    return result;
}
```

**Key math notes:**
- **FloatVec is a row vector** (`Matrix<float, 1, Dynamic>`). All rotations
  multiply the vector on the LEFT of the matrix: `vec * P` for forward,
  `vec * P^T` for inverse. The plan's original `P^T * vec` was wrong.
- **Segment centroids are stored in the rotated space** (after per-segment
  rotation was applied during encode). To reconstruct in pre-rotation
  space, apply `P^T` to both the code and the centroid.
- **Norm-matching via `o_l2norm`** preserves CAQ's magnitude invariant.
  Without this, reconstructed vectors would be unit-magnitude (since
  `v_mx=1` after rescale) rather than matching the original norm.

- [ ] **Step 7: Persist PCA state + raw codes through `save()`/`load()`**

In `IVF::save()`, after the existing serialization (after the clusters are
written), append:

```cpp
// PCA state
output.write(reinterpret_cast<const char*>(&pca_applied_), sizeof(bool));
if (pca_applied_) {
    size_t D = static_cast<size_t>(pca_mean_.cols());  // FloatVec is (1, D)
    output.write(reinterpret_cast<const char*>(&D), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(pca_mean_.data()), D * sizeof(float));
    output.write(reinterpret_cast<const char*>(pca_rotation_.data()),
                 D * D * sizeof(float));
}

// Raw codes (optional; only present if fit() was called)
bool has_raw_codes = !raw_codes_.empty();
output.write(reinterpret_cast<const char*>(&has_raw_codes), sizeof(bool));
if (has_raw_codes) {
    size_t nc = raw_codes_.size();
    output.write(reinterpret_cast<const char*>(&nc), sizeof(size_t));
    for (const auto& cluster_segs : raw_codes_) {
        size_t ns = cluster_segs.size();
        output.write(reinterpret_cast<const char*>(&ns), sizeof(size_t));
        for (const auto& m : cluster_segs) {
            size_t rows = static_cast<size_t>(m.rows());
            size_t cols = static_cast<size_t>(m.cols());
            output.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
            output.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
            output.write(reinterpret_cast<const char*>(m.data()),
                         rows * cols * sizeof(uint16_t));
        }
    }
}
```

In `IVF::load()`, after the existing deserialization, append the matching
reader:

```cpp
// PCA state
input.read(reinterpret_cast<char*>(&pca_applied_), sizeof(bool));
if (pca_applied_) {
    size_t D;
    input.read(reinterpret_cast<char*>(&D), sizeof(size_t));
    pca_mean_.resize(1, static_cast<Eigen::Index>(D));
    input.read(reinterpret_cast<char*>(pca_mean_.data()), D * sizeof(float));
    pca_rotation_.resize(static_cast<Eigen::Index>(D), static_cast<Eigen::Index>(D));
    input.read(reinterpret_cast<char*>(pca_rotation_.data()), D * D * sizeof(float));
}

// Raw codes
bool has_raw_codes = false;
input.read(reinterpret_cast<char*>(&has_raw_codes), sizeof(bool));
if (has_raw_codes) {
    size_t nc;
    input.read(reinterpret_cast<char*>(&nc), sizeof(size_t));
    raw_codes_.resize(nc);
    for (size_t c = 0; c < nc; ++c) {
        size_t ns;
        input.read(reinterpret_cast<char*>(&ns), sizeof(size_t));
        raw_codes_[c].resize(ns);
        for (size_t s = 0; s < ns; ++s) {
            size_t rows, cols;
            input.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
            input.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
            raw_codes_[c][s].resize(static_cast<Eigen::Index>(rows),
                                    static_cast<Eigen::Index>(cols));
            input.read(reinterpret_cast<char*>(raw_codes_[c][s].data()),
                       rows * cols * sizeof(uint16_t));
        }
    }
}

// Rebuild id_to_location_ from loaded cluster data
id_to_location_.clear();
id_to_location_.reserve(num_data_);
for (size_t cid = 0; cid < parallel_clusters_.size(); ++cid) {
    const PID* cluster_ids = parallel_clusters_[cid].ids();
    size_t nv = parallel_clusters_[cid].num_vec_;
    for (size_t local = 0; local < nv; ++local) {
        id_to_location_[cluster_ids[local]] = {cid, local};
    }
}
```

- [ ] **Step 8: Build and verify compilation**

```bash
export PATH=/home/rohil/miniconda3/bin:$PATH
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/saq-ivf-fit-api
cmake --build build --target saq 2>&1 | tail -20
```
Expected: zero errors. Samples target also still compiles (confirms the
`quantize()` / `quantize_cluster()` signature changes are backward-compatible
via default arguments).

---

### Task 8: C++ unit test for `fit()` and `decompress()`

**Files:**
- Create: `tests/test_fit_decompress.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create `tests/test_fit_decompress.cpp`**

```cpp
// tests/test_fit_decompress.cpp
// Smoke test: fit() on synthetic data, verify decompress() returns plausible vectors.
#include <gtest/gtest.h>
#include "index/ivf_index.h"
#include "saq/config.h"
#include <random>

using namespace saq;

static FloatRowMat make_random(int N, int D, int seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    FloatRowMat X(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            X(i, j) = nd(rng);
    // L2-normalize rows (IP use-case)
    X = X.array().colwise() / X.rowwise().norm().array();
    return X;
}

TEST(IvfFitDecompress, FitSmoke) {
    const int N = 1024, D = 64, K = 32;
    FloatRowMat X = make_random(N, D);

    QuantizeConfig cfg;
    cfg.avg_bits = 4;
    IVF ivf(N, D, K, cfg);
    // fit() should not throw
    ASSERT_NO_THROW(ivf.fit(X, /*apply_pca=*/false, K, /*seed=*/0, /*threads=*/1));
    EXPECT_EQ(ivf.num_data(), static_cast<size_t>(N));
    EXPECT_EQ(ivf.num_dim(),  static_cast<size_t>(D));
}

TEST(IvfFitDecompress, DecompressShapeAndFinite) {
    const int N = 512, D = 32, K = 16;
    FloatRowMat X = make_random(N, D, 42);

    QuantizeConfig cfg;
    cfg.avg_bits = 4;
    IVF ivf(N, D, K, cfg);
    ivf.fit(X, /*apply_pca=*/false, K, /*seed=*/0, /*threads=*/1);

    // Decompress a sample of vectors
    std::vector<PID> ids = {0, 1, 10, 100, 511};
    FloatRowMat recon = ivf.decompress(ids);

    EXPECT_EQ(recon.rows(), static_cast<Eigen::Index>(ids.size()));
    EXPECT_EQ(recon.cols(), static_cast<Eigen::Index>(D));
    // All values should be finite
    EXPECT_TRUE(recon.allFinite());
}

TEST(IvfFitDecompress, DecompressMSEReasonable_NoPCA) {
    // At 4 bpd, reconstruction MSE should be << 1 for unit-normalized vectors
    const int N = 1024, D = 64, K = 32;
    FloatRowMat X = make_random(N, D, 7);

    QuantizeConfig cfg;
    cfg.avg_bits = 4;
    IVF ivf(N, D, K, cfg);
    ivf.fit(X, /*apply_pca=*/false, K, /*seed=*/0, /*threads=*/1);

    std::vector<PID> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    FloatRowMat recon = ivf.decompress(ids);

    float mse = (X - recon).array().square().mean();
    EXPECT_LT(mse, 0.2f) << "MSE=" << mse << " — decompress likely broken (no PCA)";
}

TEST(IvfFitDecompress, DecompressMSEReasonable_WithPCA) {
    const int N = 1024, D = 64, K = 32;
    FloatRowMat X = make_random(N, D, 11);

    QuantizeConfig cfg;
    cfg.avg_bits = 4;
    IVF ivf(N, D, K, cfg);
    ivf.fit(X, /*apply_pca=*/true, K, /*seed=*/0, /*threads=*/1);

    std::vector<PID> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    FloatRowMat recon = ivf.decompress(ids);

    float mse = (X - recon).array().square().mean();
    EXPECT_LT(mse, 0.2f) << "MSE=" << mse << " — decompress with PCA failed";
}

TEST(IvfFitDecompress, DecompressThrowsAfterConstructOnly) {
    // When construct() is called directly (not fit()), raw_codes_ is empty
    // and decompress() should fail rather than return garbage.
    const int N = 64, D = 16, K = 8;
    FloatRowMat X = make_random(N, D);
    FloatRowMat centroids = X.topRows(K);  // trivial init
    std::vector<PID> assigns(N, 0);

    QuantizeConfig cfg;
    cfg.avg_bits = 4;
    IVF ivf(N, D, K, cfg);
    // Need variance to be set before construct() — normally preprocessing does this.
    FloatVec variances(1, D);
    variances.setOnes();
    ivf.set_variance(variances);
    ivf.construct(X, centroids, assigns.data(), /*num_threads=*/1);

    std::vector<PID> ids = {0};
    EXPECT_DEATH(ivf.decompress(ids), "raw codes not cached");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

- [ ] **Step 2: Register in `tests/CMakeLists.txt`**

```cmake
add_executable(test_fit_decompress test_fit_decompress.cpp)
target_link_libraries(test_fit_decompress PRIVATE saq GTest::gtest_main)
add_test(NAME test_fit_decompress COMMAND test_fit_decompress)
```

- [ ] **Step 3: Build and run tests (Linux + Faiss)**

```bash
export PATH=/home/rohil/miniconda3/bin:$PATH
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/saq-ivf-fit-api
cmake -B build \
  -DSAQ_BUILD_TESTS=ON \
  -DSAQ_BUILD_SAMPLES=OFF \
  -DSAQ_USE_OPENMP=ON \
  -DSAQ_USE_FAISS=ON \
  -DSAQ_REQUIRE_AVX512=ON \
  -Dfaiss_DIR=$CONDA_PREFIX/share/faiss \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_fit_decompress -j 4
OMP_NUM_THREADS=1 ./build/tests/test_fit_decompress 2>&1
```
Expected: `[  PASSED  ] 5 tests.`

- [ ] **Step 4: Commit Tasks 6-8 as one logical unit**

```bash
git add include/saq/quantizer.h include/saq/saq_quantizer.h \
        include/index/ivf_index.h src/ivf_index.cpp \
        tests/test_fit_decompress.cpp tests/CMakeLists.txt
git commit -m "feat(ivf): add IVF::fit() and IVF::decompress() with raw-code caching"
```

---

### Task 9: Pybind11 binding updates

**Files:**
- Modify: `python/bindings/saq_bindings.cpp`

- [ ] **Step 1: Add `fit`, `decompress`, `set_codebooks`, `set_gaussian_codebooks` to the IVF pybind class**

In `saq_bindings.cpp`, append these `.def()` calls to the `py::class_<IVF>` block (before the closing semicolon):

```cpp
.def("fit",
     [](IVF &self, Eigen::Ref<const FloatRowMat> X,
        bool apply_pca, int K, int seed, int num_threads) {
         py::gil_scoped_release release;
         self.fit(X, apply_pca, K, seed, num_threads);
     },
     py::arg("X"),
     py::arg("apply_pca") = true,
     py::arg("K") = 4096,
     py::arg("seed") = 0,
     py::arg("num_threads") = 8,
     "Self-contained preprocessing + construction. Handles PCA and k-means internally.")

.def("decompress",
     [](const IVF &self, py::array_t<uint32_t> ids) -> Eigen::MatrixXf {
         auto buf = ids.request();
         if (buf.ndim != 1)
             throw std::runtime_error("ids must be a 1D uint32 array");
         const PID *ptr = static_cast<const PID *>(buf.ptr);
         std::vector<PID> id_vec(ptr, ptr + buf.size);
         py::gil_scoped_release release;
         return self.decompress(id_vec);
     },
     py::arg("ids"),
     "Reconstruct approximate vectors. ids: 1D uint32 array. Returns (N, D) float32.")

// Codebook methods — identical source across all branches; runtime-guarded by macro.
.def("set_codebooks",
     [](IVF& /*self*/, Eigen::Ref<const FloatRowMat> /*codebooks*/) {
#ifdef SAQ_ENABLE_CODEBOOK
         // self.set_codebooks(codebooks);  // uncomment on codebook branch
         throw std::runtime_error("set_codebooks: not yet wired on this branch");
#else
         throw std::runtime_error(
             "This SAQ build does not support set_codebooks. "
             "Install the saq-codebook wheel.");
#endif
     },
     py::arg("codebooks"),
     "Set precomputed Gaussian codebook. Available on codebook branches only.")

.def("set_gaussian_codebooks",
     [](IVF& /*self*/, Eigen::Ref<const FloatRowMat> /*base*/,
        Eigen::Ref<const FloatVec> /*stds*/) {
#ifdef SAQ_ENABLE_CODEBOOK
         // self.set_gaussian_codebooks(base, stds);
         throw std::runtime_error("set_gaussian_codebooks: not yet wired on this branch");
#else
         throw std::runtime_error(
             "This SAQ build does not support set_gaussian_codebooks. "
             "Install the saq-codebook wheel.");
#endif
     },
     py::arg("base_codebook"), py::arg("residual_stds"),
     "Set Gaussian codebook with per-segment residual stds. Codebook branches only.")
```

- [ ] **Step 2: Build Python bindings**

```bash
cmd.exe /c "cmake -B build_python -G Ninja -DCMAKE_BUILD_TYPE=Release -DSAQ_BUILD_PYTHON=ON -DSAQ_USE_OPENMP=ON -DSAQ_USE_FAISS=OFF -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl && cmake --build build_python --target _saq_core 2>&1 | findstr /i error"
```
Expected: zero error lines, `_saq_core.pyd` produced.

- [ ] **Step 3: Smoke test bindings from Python**

```python
import sys
sys.path.insert(0, 'python')
sys.path.insert(0, 'build_python/python/bindings/Release')
import saq
import numpy as np

X = np.random.randn(512, 32).astype(np.float32)
cfg = saq.QuantizeConfig()
cfg.avg_bits = 4.0
ivf = saq.IVF(512, 32, 16, cfg)
ivf.fit(X, apply_pca=False, K=16, seed=0, num_threads=1)
print("num_data:", ivf.num_data)
ids = np.arange(10, dtype=np.uint32)
recon = ivf.decompress(ids)
print("recon shape:", recon.shape)   # expect (10, 32)
assert recon.shape == (10, 32)
print("PASS")
```

Save as `/tmp/test_bindings.py` and run:
```bash
cmd.exe /c "python /tmp/test_bindings.py"
```
Expected: `PASS`.

- [ ] **Step 4: Commit**

```bash
git add python/bindings/saq_bindings.cpp
git commit -m "feat: add fit/decompress/set_codebooks bindings to IVF"
```

---

### Task 10: Linux verification with Faiss (on PACE or local Linux)

**Files:** No new files — build verification only.

- [ ] **Step 1: Build with `SAQ_USE_FAISS=ON`**

On a Linux machine with `faiss-cpu` installed (e.g., `conda install -c pytorch faiss-cpu`):
```bash
cmake -B build_linux -DSAQ_BUILD_PYTHON=ON -DSAQ_USE_FAISS=ON \
      -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build_linux --target _saq_core -j8
```
Expected: builds without errors.

- [ ] **Step 2: Run the same Python smoke test on Linux**

```bash
python /tmp/test_bindings.py
```
Expected: `PASS`.

---

## Phase 2 — Branch Propagation

### Task 11: Merge `feat/ivf-fit-api` → `main`, then propagate

**Files:** Git operations only.

- [ ] **Step 1: Open PR and merge `feat/ivf-fit-api` → `main`**

```bash
git checkout feat/ivf-fit-api
git push origin feat/ivf-fit-api
# Open PR via GitHub UI or gh CLI
gh pr create --title "feat: add preprocessing module, IVF::fit(), decompress(), SAQ_USE_FAISS" \
             --body "See design doc: docs/superpowers/specs/2026-04-10-saq-python-packaging-and-test-bench-design.md" \
             --base main
# After review:
gh pr merge --squash
```

- [ ] **Step 2: Propagate to `gpu` branch**

```bash
git checkout gpu
git merge main
# Resolve conflicts if any (GPU-specific files shouldn't conflict)
git push origin gpu
```

- [ ] **Step 3: Propagate to `feat/optimal-codebook` branch**

```bash
git checkout feat/optimal-codebook
git merge main
git push origin feat/optimal-codebook
```

- [ ] **Step 4: Create `gpu-codebook` stub branch from `gpu`**

```bash
git checkout gpu
git checkout -b gpu-codebook
```

- [ ] **Step 5: Wire codebook stub on `gpu-codebook`**

In `python/bindings/saq_bindings.cpp` on this branch, update `set_codebooks` and `set_gaussian_codebooks` to raise `NotImplementedError` with clear message (GPU kernels don't support codebook lookup yet). The `#ifndef SAQ_ENABLE_CODEBOOK` guard already handles this — just add `-DSAQ_ENABLE_CODEBOOK` to the CMake build for this branch to activate the guard branch, but leave the actual call commented out:

```cmake
# In CMakeLists.txt on gpu-codebook branch, add:
target_compile_definitions(saq PRIVATE SAQ_ENABLE_CODEBOOK)
```

The binding already throws `"not yet wired on this branch"` inside `#ifdef SAQ_ENABLE_CODEBOOK` — this is the correct stub behavior.

- [ ] **Step 6: Commit and push `gpu-codebook`**

```bash
git add CMakeLists.txt
git commit -m "feat(gpu-codebook): stub branch with codebook API surface, NotImplementedError at runtime"
git push origin gpu-codebook
```

- [ ] **Step 7: Verify all 4 wheels build and expose identical API**

For each branch: `main`, `gpu`, `feat/optimal-codebook`, `gpu-codebook`, run:
```bash
cmake --build build_python --target _saq_core
python -c "import saq; ivf = saq.IVF(); print(dir(ivf))" 2>&1 | grep -E "fit|decompress|set_codebooks"
```
Expected for all four: `fit`, `decompress`, `set_codebooks`, `set_gaussian_codebooks` present in `dir(ivf)`.

---

## Phase 3 — Test Bench: `BaseSearchIndex` ABC and `SaqIndex`

### Task 12: `BaseSearchIndex` ABC

**Files:**
- Create: `src/haag_vq/methods/base_search_index.py`

- [ ] **Step 1: Create `base_search_index.py`**

```python
# src/haag_vq/methods/base_search_index.py
"""Primary benchmark interface for all VQ search methods.

Every method (SAQ, PQ, OPQ, SQ, RaBitQ, Faiss baselines) implements this
interface — either natively (SAQ, Faiss) or via wrapper classes that adapt
BaseQuantizer implementations (FlatQuantizedIndex, IvfQuantizedIndex).

Primary metrics : recall@k, QPS, memory_footprint
Secondary metric: reconstruction MSE (optional, via reconstruction_mse())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np


class BaseSearchIndex(ABC):
    """Abstract base class for all VQ search methods benchmarked in haag_vq."""

    @abstractmethod
    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        """Learn index from training vectors X of shape (N, D).

        All preprocessing (PCA, k-means, codebook fitting) is internal.
        After fit(), search() and memory_footprint() must be valid.
        """

    @abstractmethod
    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        """Return (nq, k) uint32 array of approximate nearest neighbor IDs.

        Q: (nq, D) float32 query matrix.
        k: number of neighbors.
        """

    @abstractmethod
    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) where both are (nq, k) float32 / uint32.

        For 'l2' metric: distances are squared L2.
        For 'ip' metric: distances are inner products (higher = better).
        """

    @abstractmethod
    def memory_footprint(self) -> int:
        """Estimated index memory in bytes.

        Used for compression ratio computation.
        Does not include Python object overhead — only the encoded data
        and auxiliary structures (centroids, codebooks, etc.).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist index to disk at the given path."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore index from disk."""

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Return mean reconstruction MSE over a sample of vectors.

        This is a secondary benchmark metric. The default implementation
        returns None (method does not support reconstruction).

        Concrete classes override this when their underlying quantizer or
        index supports decompress(). Used as a secondary benchmark column
        alongside recall@k, QPS, and memory_footprint.

        Args:
            X:          Original vectors (N, D) used as ground truth.
            sample_ids: If provided, compute MSE over X[sample_ids].
                        If None, use all vectors.

        Returns:
            Mean per-element MSE as float, or None if not supported.
        """
        return None
```

- [ ] **Step 2: Verify `BaseSearchIndex` can be imported and subclassed**

```python
# /tmp/test_abc.py
import sys
sys.path.insert(0, 'src')
from haag_vq.methods.base_search_index import BaseSearchIndex
import numpy as np
from typing import Tuple

class DummyIndex(BaseSearchIndex):
    def fit(self, X, metric='l2'): self._N = len(X)
    def search(self, Q, k): return np.zeros((len(Q), k), dtype=np.uint32)
    def search_with_scores(self, Q, k):
        ids = np.zeros((len(Q), k), dtype=np.uint32)
        dists = np.zeros((len(Q), k), dtype=np.float32)
        return ids, dists
    def memory_footprint(self): return 0
    def save(self, path): pass
    def load(self, path): pass

idx = DummyIndex()
idx.fit(np.zeros((10, 4), dtype=np.float32))
assert idx.reconstruction_mse(np.zeros((10, 4))) is None
print("PASS")
```

```bash
python /tmp/test_abc.py
```
Expected: `PASS`.

---

### Task 13: `SaqIndex` wrapper

**Files:**
- Create: `src/haag_vq/methods/search/__init__.py`
- Create: `src/haag_vq/methods/search/saq_index.py`

- [ ] **Step 1: Create `search/__init__.py`**

```python
# src/haag_vq/methods/search/__init__.py
from .saq_index import SaqIndex
from .flat_quantized_index import FlatQuantizedIndex
from .ivf_quantized_index import IvfQuantizedIndex
from .faiss_ivfpq_index import FaissIvfPqIndex

__all__ = [
    "SaqIndex",
    "FlatQuantizedIndex",
    "IvfQuantizedIndex",
    "FaissIvfPqIndex",
]
```

- [ ] **Step 2: Create `saq_index.py`**

```python
# src/haag_vq/methods/search/saq_index.py
"""SaqIndex — wraps saq.IVF (or saq.GpuIVF) as a BaseSearchIndex."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class SaqIndex(BaseSearchIndex):
    """Wraps the SAQ C++ IVF index as a BaseSearchIndex.

    Capability detection: imports saq at construction time and checks for
    GpuIVF. Variant selection happens at wheel install time — install
    saq-cpu, saq-gpu, saq-codebook, or saq-gpu-codebook.
    """

    def __init__(
        self,
        bpd: float = 4.0,
        K: int = 4096,
        nprobe: int = 200,
        use_gpu: bool = False,
        use_codebook: bool = False,
        num_threads: int = 8,
    ) -> None:
        import saq as _saq
        self._saq = _saq
        self._bpd = bpd
        self._K = K
        self._nprobe = nprobe
        self._use_gpu = use_gpu and hasattr(_saq, 'GpuIVF')
        self._use_codebook = use_codebook
        self._num_threads = num_threads
        self._index = None
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def _make_config(self, metric: str):
        cfg = self._saq.QuantizeConfig()
        cfg.avg_bits = self._bpd
        if metric == 'ip':
            cfg.single.quant_type = self._saq.BaseQuantType.CAQ
        cfg.single.random_rotation = True
        cfg.enable_segmentation = True
        return cfg

    def _make_searcher_config(self):
        scfg = self._saq.SearcherConfig()
        scfg.dist_type = (
            self._saq.DistType.IP if self._metric == 'ip' else self._saq.DistType.L2Sqr
        )
        return scfg

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape
        cfg = self._make_config(metric)
        IndexClass = self._saq.GpuIVF if self._use_gpu else self._saq.IVF
        self._index = IndexClass(self._N, self._D, self._K, cfg)
        if self._use_codebook:
            # Codebook must be set before fit() on codebook-enabled builds.
            # Users call set_codebooks() after construction and before fit().
            pass
        self._index.fit(X, apply_pca=True, K=self._K, seed=0,
                        num_threads=self._num_threads)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        scfg = self._make_searcher_config()
        ids = self._index.search_batch(Q, k, self._nprobe, scfg)
        # search_batch returns uint32 IDs only; scores require per-query search
        # Return dummy distances (0.0) for now — use search() path for recall.
        dists = np.zeros((Q.shape[0], k), dtype=np.float32)
        return ids, dists

    def memory_footprint(self) -> int:
        if self._index is None:
            return 0
        # Approximate: N * bpd / 8 bytes for codes + centroid storage
        code_bytes = int(self._N * self._bpd / 8.0)
        centroid_bytes = self._K * self._D * 4  # float32
        return code_bytes + centroid_bytes

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("SaqIndex.save() called before fit()")
        self._index.save(str(path))

    def load(self, path: str | Path) -> None:
        self._index = self._saq.IVF()
        self._index.load(str(path))

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._index is None:
            return None
        X = np.ascontiguousarray(X, dtype=np.float32)
        if sample_ids is None:
            sample_ids = np.arange(len(X), dtype=np.uint32)
        sample_ids = np.ascontiguousarray(sample_ids, dtype=np.uint32)
        X_hat = self._index.decompress(sample_ids)
        return float(np.mean((X[sample_ids] - X_hat) ** 2))
```

- [ ] **Step 3: Write failing test for `SaqIndex` contract**

```python
# tests/test_saq_index.py  (rewrite of test_saq.py)
import pytest
import numpy as np

def make_data(N=512, D=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

@pytest.fixture
def saq_index():
    pytest.importorskip("saq")  # skip if SAQ wheel not installed
    from haag_vq.methods.search.saq_index import SaqIndex
    return SaqIndex(bpd=4.0, K=16, nprobe=8, num_threads=1)

def test_fit_does_not_raise(saq_index):
    X = make_data()
    saq_index.fit(X, metric='l2')

def test_search_shape(saq_index):
    X = make_data()
    saq_index.fit(X)
    Q = make_data(N=10, seed=99)
    ids = saq_index.search(Q, k=5)
    assert ids.shape == (10, 5)
    assert ids.dtype == np.uint32

def test_search_with_scores_shape(saq_index):
    X = make_data()
    saq_index.fit(X)
    Q = make_data(N=5, seed=42)
    ids, dists = saq_index.search_with_scores(Q, k=3)
    assert ids.shape == (5, 3)
    assert dists.shape == (5, 3)

def test_memory_footprint_positive(saq_index):
    X = make_data()
    saq_index.fit(X)
    mem = saq_index.memory_footprint()
    assert mem > 0

def test_reconstruction_mse_finite(saq_index):
    X = make_data()
    saq_index.fit(X)
    mse = saq_index.reconstruction_mse(X)
    assert mse is not None
    assert np.isfinite(mse)
    # Very loose: just not NaN/inf
    assert mse >= 0.0

def test_reconstruction_mse_with_sample_ids(saq_index):
    X = make_data()
    saq_index.fit(X)
    sample_ids = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
    mse = saq_index.reconstruction_mse(X, sample_ids=sample_ids)
    assert mse is not None and np.isfinite(mse)

def test_save_load_roundtrip(saq_index, tmp_path):
    X = make_data()
    saq_index.fit(X)
    p = tmp_path / "saq_test.idx"
    saq_index.save(p)
    from haag_vq.methods.search.saq_index import SaqIndex
    loaded = SaqIndex(bpd=4.0, K=16, nprobe=8, num_threads=1)
    loaded.load(p)
    Q = make_data(N=5, seed=7)
    ids_orig   = saq_index.search(Q, k=3)
    ids_loaded = loaded.search(Q, k=3)
    assert np.array_equal(ids_orig, ids_loaded)
```

- [ ] **Step 4: Run test (will fail until `saq` wheel installed or available)**

```bash
cd /path/to/vector-quantization
pytest tests/test_saq_index.py -v 2>&1 | head -40
```
Expected: `SKIPPED` (if no `saq` wheel) or `PASSED` (if wheel installed).

---

## Phase 4 — Test Bench: Quantizer Wrappers

### Task 14: `FlatQuantizedIndex`

**Files:**
- Create: `src/haag_vq/methods/search/flat_quantized_index.py`
- Create: `tests/test_flat_quantized.py`

- [ ] **Step 1: Create `flat_quantized_index.py`**

```python
# src/haag_vq/methods/search/flat_quantized_index.py
"""FlatQuantizedIndex — wraps any BaseQuantizer with brute-force search."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from haag_vq.methods.base_search_index import BaseSearchIndex
from haag_vq.methods.base_quantizer import BaseQuantizer


class FlatQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer with brute-force search.

    Fit: compresses all training vectors into codes.
    Search: decompresses stored codes, computes exact distance, top-k.

    Complexity: O(N * D) per query — fair baseline but slow for large N.
    Use IvfQuantizedIndex for faster search at comparable compression.
    """

    def __init__(self, quantizer: BaseQuantizer) -> None:
        self._quantizer = quantizer
        self._codes: Optional[np.ndarray] = None
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape
        self._quantizer.fit(X)
        self._codes = self._quantizer.compress(X)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        X_hat = self._quantizer.decompress(self._codes).astype(np.float32)
        nq = Q.shape[0]
        k = min(k, self._N)

        if self._metric == 'l2':
            dists = cdist(Q, X_hat, metric='sqeuclidean').astype(np.float32)
            top_k = np.argpartition(dists, k - 1, axis=1)[:, :k]
            # Sort within top-k
            top_k_sorted = np.array([
                top_k[i][np.argsort(dists[i][top_k[i]])]
                for i in range(nq)
            ])
        else:  # 'ip'
            sims = (Q @ X_hat.T).astype(np.float32)
            top_k = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_k_sorted = np.array([
                top_k[i][np.argsort(-sims[i][top_k[i]])]
                for i in range(nq)
            ])
            dists = sims

        gathered = np.take_along_axis(
            dists, top_k_sorted, axis=1
        )
        return top_k_sorted.astype(np.uint32), gathered

    def memory_footprint(self) -> int:
        return int(self._codes.nbytes) if self._codes is not None else 0

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._codes is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(self._N)
        X_hat = self._quantizer.decompress(self._codes[sample_ids]).astype(np.float32)
        return float(np.mean((X[sample_ids].astype(np.float32) - X_hat) ** 2))

    def save(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'quantizer': self._quantizer, 'codes': self._codes,
                         'metric': self._metric, 'N': self._N, 'D': self._D}, f)

    def load(self, path: str | Path) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._quantizer = state['quantizer']
        self._codes     = state['codes']
        self._metric    = state['metric']
        self._N         = state['N']
        self._D         = state['D']
```

- [ ] **Step 2: Create `tests/test_flat_quantized.py`**

```python
# tests/test_flat_quantized.py
import pytest
import numpy as np

def make_data(N=256, D=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D)).astype(np.float32)

@pytest.fixture
def flat_pq():
    from haag_vq.methods.base_quantizer import ProductQuantizer
    from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
    return FlatQuantizedIndex(ProductQuantizer(M=4, Ks=16))

def test_fit_search_shape(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    Q = make_data(N=5, seed=42)
    ids = flat_pq.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32

def test_search_with_scores_shape(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    Q = make_data(N=3, seed=7)
    ids, dists = flat_pq.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert dists.shape == (3, 4)

def test_memory_footprint(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    assert flat_pq.memory_footprint() > 0

def test_reconstruction_mse(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    mse = flat_pq.reconstruction_mse(X)
    assert mse is not None
    assert np.isfinite(mse) and mse >= 0.0

def test_save_load(flat_pq, tmp_path):
    X = make_data()
    flat_pq.fit(X)
    p = tmp_path / "flat.pkl"
    flat_pq.save(p)
    from haag_vq.methods.base_quantizer import ProductQuantizer
    from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
    loaded = FlatQuantizedIndex(ProductQuantizer(M=4, Ks=16))
    loaded.load(p)
    Q = make_data(N=5, seed=1)
    assert np.array_equal(flat_pq.search(Q, k=3), loaded.search(Q, k=3))
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_flat_quantized.py -v
```
Expected: all tests PASS.

---

### Task 15: `IvfQuantizedIndex`

**Files:**
- Create: `src/haag_vq/methods/search/ivf_quantized_index.py`
- Create: `tests/test_ivf_quantized.py`

- [ ] **Step 1: Create `ivf_quantized_index.py`**

```python
# src/haag_vq/methods/search/ivf_quantized_index.py
"""IvfQuantizedIndex — wraps any BaseQuantizer inside a k-means IVF shell."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex
from haag_vq.methods.base_quantizer import BaseQuantizer


class IvfQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer inside a k-means IVF shell.

    Fit: k-means (K clusters) + per-cluster residual quantization.
    Search: find nprobe nearest centroids, search within each cluster,
            decompress candidates, compute exact distance, top-k.

    This is a fair comparison point for SAQ: both methods use IVF +
    per-cluster quantization. The difference is in the quantizer itself.
    """

    def __init__(
        self,
        quantizer_factory: Callable[[], BaseQuantizer],
        K: int = 4096,
        nprobe: int = 200,
    ) -> None:
        self._quantizer_factory = quantizer_factory
        self._K = K
        self._nprobe = nprobe
        self._centroids: Optional[np.ndarray] = None
        self._cluster_quantizers: list[Optional[BaseQuantizer]] = []
        self._cluster_codes: list[np.ndarray] = []
        self._cluster_ids: list[np.ndarray] = []
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape

        km = faiss.Kmeans(self._D, self._K, seed=0, verbose=False)
        km.train(X)
        self._centroids = km.centroids.copy()

        _, raw_assign = km.index.search(X, 1)
        assignments = raw_assign.ravel().astype(np.int32)

        self._cluster_quantizers = []
        self._cluster_codes = []
        self._cluster_ids = []

        for c in range(self._K):
            mask = assignments == c
            vid = np.where(mask)[0].astype(np.uint32)
            if not mask.any():
                self._cluster_quantizers.append(None)
                self._cluster_codes.append(np.array([], dtype=np.uint8))
                self._cluster_ids.append(vid)
                continue
            residuals = X[mask] - self._centroids[c]
            q = self._quantizer_factory()
            q.fit(residuals)
            self._cluster_quantizers.append(q)
            self._cluster_codes.append(q.compress(residuals))
            self._cluster_ids.append(vid)

    def _nearest_centroids(self, Q: np.ndarray) -> np.ndarray:
        """Return (nq, nprobe) centroid indices sorted by distance."""
        import faiss
        index = faiss.IndexFlatL2(self._D)
        index.add(self._centroids)
        _, ids = index.search(Q, min(self._nprobe, self._K))
        return ids  # (nq, nprobe)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        nq = Q.shape[0]
        probe_ids = self._nearest_centroids(Q)  # (nq, nprobe)

        all_ids   = np.full((nq, k), -1, dtype=np.int64)
        all_dists = np.full((nq, k), np.inf, dtype=np.float32)

        for qi in range(nq):
            candidates_id   = []
            candidates_vec  = []
            for cid in probe_ids[qi]:
                q = self._cluster_quantizers[cid]
                if q is None:
                    continue
                codes = self._cluster_codes[cid]
                recon = q.decompress(codes).astype(np.float32) + self._centroids[cid]
                vids  = self._cluster_ids[cid]
                candidates_id.append(vids)
                candidates_vec.append(recon)
            if not candidates_id:
                continue
            vids = np.concatenate(candidates_id)
            vecs = np.concatenate(candidates_vec, axis=0)

            if self._metric == 'l2':
                dists = np.sum((Q[qi] - vecs) ** 2, axis=1)
            else:
                dists = -(Q[qi] @ vecs.T)

            topk_local = min(k, len(dists))
            idx = np.argpartition(dists, topk_local - 1)[:topk_local]
            idx_sorted = idx[np.argsort(dists[idx])]

            all_ids[qi, :topk_local]   = vids[idx_sorted]
            all_dists[qi, :topk_local] = dists[idx_sorted]

        return all_ids.astype(np.uint32), all_dists

    def memory_footprint(self) -> int:
        centroid_bytes = (self._centroids.nbytes
                         if self._centroids is not None else 0)
        code_bytes = sum(c.nbytes for c in self._cluster_codes)
        return centroid_bytes + code_bytes

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._centroids is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(self._N)

        # Build reverse lookup: global id -> (cluster_id, local_idx)
        # This is O(N) but only done once at benchmark time.
        id_to_loc: dict[int, tuple[int, int]] = {}
        for c in range(self._K):
            for local, gid in enumerate(self._cluster_ids[c]):
                id_to_loc[int(gid)] = (c, local)

        mse_sum = 0.0
        count = 0
        for gid in sample_ids:
            if int(gid) not in id_to_loc:
                continue
            c, local = id_to_loc[int(gid)]
            q = self._cluster_quantizers[c]
            if q is None:
                continue
            code = self._cluster_codes[c][local:local+1]
            x_hat = q.decompress(code)[0].astype(np.float32) + self._centroids[c]
            mse_sum += float(np.mean((X[gid].astype(np.float32) - x_hat) ** 2))
            count += 1
        return mse_sum / count if count > 0 else None

    def save(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'quantizer_factory': self._quantizer_factory,
                'K': self._K, 'nprobe': self._nprobe,
                'centroids': self._centroids,
                'cluster_quantizers': self._cluster_quantizers,
                'cluster_codes': self._cluster_codes,
                'cluster_ids': self._cluster_ids,
                'metric': self._metric, 'N': self._N, 'D': self._D,
            }, f)

    def load(self, path: str | Path) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        for k, v in state.items():
            setattr(self, f'_{k}', v)
```

- [ ] **Step 2: Create `tests/test_ivf_quantized.py`**

```python
# tests/test_ivf_quantized.py
import pytest
import numpy as np

def make_data(N=256, D=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D)).astype(np.float32)

@pytest.fixture
def ivf_pq():
    pytest.importorskip("faiss")
    from haag_vq.methods.base_quantizer import ProductQuantizer
    from haag_vq.methods.search.ivf_quantized_index import IvfQuantizedIndex
    return IvfQuantizedIndex(
        quantizer_factory=lambda: ProductQuantizer(M=4, Ks=8),
        K=8, nprobe=4
    )

def test_fit_search_shape(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    Q = make_data(N=5, seed=42)
    ids = ivf_pq.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32

def test_search_with_scores_shape(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    Q = make_data(N=3)
    ids, dists = ivf_pq.search_with_scores(Q, k=3)
    assert ids.shape == (3, 3)
    assert dists.shape == (3, 3)

def test_memory_footprint(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    assert ivf_pq.memory_footprint() > 0

def test_reconstruction_mse(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    mse = ivf_pq.reconstruction_mse(X, sample_ids=np.arange(10, dtype=np.uint32))
    assert mse is not None and np.isfinite(mse) and mse >= 0.0
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_ivf_quantized.py -v
```
Expected: all tests PASS.

---

### Task 16: `FaissIvfPqIndex`

**Files:**
- Create: `src/haag_vq/methods/search/faiss_ivfpq_index.py`

- [ ] **Step 1: Create `faiss_ivfpq_index.py`**

```python
# src/haag_vq/methods/search/faiss_ivfpq_index.py
"""FaissIvfPqIndex — external baseline wrapping faiss.IndexIVFPQ."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class FaissIvfPqIndex(BaseSearchIndex):
    """External baseline: faiss.IndexIVFPQ.

    Reference point: if SAQ doesn't beat IVFPQ on recall@k at the same
    bits-per-dim, the algorithm isn't pulling its weight.
    """

    def __init__(
        self,
        K: int = 4096,
        m: int = 16,    # number of PQ subspaces
        nbits: int = 8, # bits per subspace code
        nprobe: int = 200,
    ) -> None:
        import faiss as _faiss
        self._faiss = _faiss
        self._K = K
        self._m = m
        self._nbits = nbits
        self._nprobe = nprobe
        self._index = None
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._N, self._D = X.shape
        self._metric = metric

        if metric == 'ip':
            quantizer = self._faiss.IndexFlatIP(self._D)
            self._index = self._faiss.IndexIVFPQ(
                quantizer, self._D, self._K, self._m, self._nbits,
                self._faiss.METRIC_INNER_PRODUCT
            )
        else:
            quantizer = self._faiss.IndexFlatL2(self._D)
            self._index = self._faiss.IndexIVFPQ(
                quantizer, self._D, self._K, self._m, self._nbits
            )
        self._index.train(X)
        self._index.add(X)
        self._index.nprobe = self._nprobe

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        _, ids = self._index.search(Q, k)
        return ids.astype(np.uint32)

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        dists, ids = self._index.search(Q, k)
        return ids.astype(np.uint32), dists.astype(np.float32)

    def memory_footprint(self) -> int:
        if self._index is None:
            return 0
        # Centroids + PQ codes (N * m bytes) + codebook (K * m * Ks * 4 bytes)
        centroid_bytes = self._K * self._D * 4
        code_bytes     = self._N * self._m
        codebook_bytes = self._K * self._m * (1 << self._nbits) * 4
        return centroid_bytes + code_bytes + codebook_bytes

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        # IndexIVFPQ.reconstruct() exists but is not reliable across all Faiss
        # builds. Return None — document as limitation.
        return None

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        self._index = self._faiss.read_index(str(path))
```

- [ ] **Step 2: Add `FaissIvfPqIndex` to the test for contract compliance**

Append to `tests/test_flat_quantized.py` or create `tests/test_faiss_ivfpq.py`:

```python
# tests/test_faiss_ivfpq.py
import pytest
import numpy as np

def make_data(N=256, D=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D)).astype(np.float32)

@pytest.fixture
def faiss_idx():
    pytest.importorskip("faiss")
    from haag_vq.methods.search.faiss_ivfpq_index import FaissIvfPqIndex
    return FaissIvfPqIndex(K=8, m=4, nbits=4, nprobe=4)

def test_fit_search_shape(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    Q = make_data(N=5, seed=42)
    ids = faiss_idx.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32

def test_reconstruction_mse_none(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    assert faiss_idx.reconstruction_mse(X) is None
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_faiss_ivfpq.py -v
```
Expected: all tests PASS.

---

## Phase 5 — Test Bench: Benchmark Harness

### Task 17: `search_bench.py`

**Files:**
- Create: `src/haag_vq/benchmarks/search_bench.py`

- [ ] **Step 1: Create `search_bench.py`**

```python
# src/haag_vq/benchmarks/search_bench.py
"""Unified benchmark harness for BaseSearchIndex implementations.

Primary metrics : recall@k, QPS, memory_footprint
Secondary metric: reconstruction MSE
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from haag_vq.methods.base_search_index import BaseSearchIndex


def compute_recall(ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k.

    ids:          (nq, k_returned) uint32 — returned neighbor IDs
    ground_truth: (nq, k_gt) uint32 — true neighbor IDs (sorted by distance)
    k:            recall cutoff

    Returns fraction of queries where at least one true top-k neighbor
    appears in the returned IDs.
    """
    nq = ids.shape[0]
    k_gt = min(k, ground_truth.shape[1])
    k_ret = min(k, ids.shape[1])
    hits = 0
    for i in range(nq):
        gt_set = set(ground_truth[i, :k_gt].tolist())
        ret_set = set(ids[i, :k_ret].tolist())
        if gt_set & ret_set:
            hits += 1
    return hits / nq


def benchmark_index(
    index: BaseSearchIndex,
    X: np.ndarray,
    Q: np.ndarray,
    ground_truth: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
    mse_sample: int = 1000,
    repeats: int = 3,
    index_params: Optional[dict] = None,
) -> dict:
    """Benchmark a BaseSearchIndex on recall@k, QPS, memory, and MSE.

    The index must already be fit() before calling this function.
    build_time_s should be measured externally and passed via index_params.

    Returns dict with keys:
        method, params, build_time_s, qps, memory_bytes, memory_mb,
        recall@1, recall@10, recall@100, recon_mse
    """
    results: dict = {
        'method': type(index).__name__,
        'params': index_params or {},
    }

    # --- Search timing ---
    nq = Q.shape[0]
    k_max = max(ks)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ids = index.search(Q, k_max)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best_time = min(times)
    results['qps'] = nq / best_time if best_time > 0 else float('inf')

    # --- Recall ---
    for k in ks:
        results[f'recall@{k}'] = compute_recall(ids, ground_truth, k)

    # --- Memory ---
    mem = index.memory_footprint()
    results['memory_bytes'] = mem
    results['memory_mb'] = mem / (1024 ** 2)

    # --- Secondary: reconstruction MSE ---
    n_total = len(X)
    if n_total > mse_sample:
        rng = np.random.default_rng(0)
        sample_ids = rng.choice(n_total, mse_sample, replace=False).astype(np.uint32)
    else:
        sample_ids = np.arange(n_total, dtype=np.uint32)

    results['recon_mse'] = index.reconstruction_mse(X, sample_ids=sample_ids)

    return results


def sweep_bpd(
    IndexClass: type,
    bpd_values: list[float],
    X: np.ndarray,
    Q: np.ndarray,
    ground_truth: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
    **index_kwargs,
) -> list[dict]:
    """Sweep bits-per-dim values, fit and benchmark each, return results list.

    Assumes IndexClass.__init__ accepts bpd as the first keyword argument.
    All other index_kwargs are forwarded as-is.
    """
    all_results = []
    for bpd in bpd_values:
        index = IndexClass(bpd=bpd, **index_kwargs)
        t0 = time.perf_counter()
        index.fit(X)
        build_time = time.perf_counter() - t0
        result = benchmark_index(index, X, Q, ground_truth, ks=ks,
                                 index_params={'bpd': bpd, **index_kwargs})
        result['build_time_s'] = build_time
        all_results.append(result)
    return all_results


def compare_methods(
    indices: list[BaseSearchIndex],
    X: np.ndarray,
    Q: np.ndarray,
    ground_truth: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
    build_times: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Run all methods (already fit), return DataFrame comparing metrics.

    Columns: method, recall@k (one per k), qps, memory_mb, recon_mse
    """
    rows = []
    for i, idx in enumerate(indices):
        result = benchmark_index(idx, X, Q, ground_truth, ks=ks)
        if build_times is not None:
            result['build_time_s'] = build_times[i]
        rows.append(result)
    return pd.DataFrame(rows)


def pareto_plot(
    results: list[dict],
    x: str = 'memory_mb',
    y: str = 'recall@10',
    label_col: str = 'method',
) -> None:
    """Plot Pareto curve of benchmark results.

    x: 'memory_mb' | 'qps' | 'memory_bytes'
    y: 'recall@1' | 'recall@10' | 'recall@100' | 'recon_mse'
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.scatter(r.get(x), r.get(y), label=r.get(label_col, '?'), s=80)
        ax.annotate(r.get(label_col, ''), (r.get(x), r.get(y)),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Pareto: {y} vs {x}')
    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 2: Smoke test `search_bench.py`**

```python
# /tmp/test_bench.py
import sys; sys.path.insert(0, 'src')
import numpy as np
from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
from haag_vq.methods.base_quantizer import ProductQuantizer
from haag_vq.benchmarks.search_bench import benchmark_index, compare_methods

rng = np.random.default_rng(0)
X  = rng.standard_normal((256, 16)).astype(np.float32)
Q  = rng.standard_normal((20, 16)).astype(np.float32)
gt = np.zeros((20, 10), dtype=np.uint32)  # dummy gt

idx = FlatQuantizedIndex(ProductQuantizer(M=4, Ks=8))
idx.fit(X)
result = benchmark_index(idx, X, Q, gt, ks=(1, 10))
print("qps:", result['qps'])
print("recall@1:", result['recall@1'])
print("memory_mb:", result['memory_mb'])
print("recon_mse:", result['recon_mse'])
print("PASS")
```

```bash
python /tmp/test_bench.py
```
Expected: printed values, `PASS`.

---

### Task 18: Rewrite `run_benchmarks.py`

**Files:**
- Modify: `src/haag_vq/benchmarks/run_benchmarks.py`

- [ ] **Step 1: Rewrite `run_benchmarks.py` to dispatch via `BaseSearchIndex`**

```python
# src/haag_vq/benchmarks/run_benchmarks.py
"""Benchmark runner — dispatches all VQ methods through BaseSearchIndex."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.search_bench import benchmark_index, pareto_plot


def load_dataset(data_dir: Path):
    """Load X (train), Q (queries), gt (ground truth) from a data directory.

    Supports .fvecs / .ivecs format via numpy.  Adjust for your dataset layout.
    """
    X  = np.load(data_dir / 'train.npy').astype(np.float32)
    Q  = np.load(data_dir / 'queries.npy').astype(np.float32)
    gt = np.load(data_dir / 'groundtruth.npy').astype(np.uint32)
    return X, Q, gt


def build_methods(args) -> list:
    """Instantiate all configured benchmark methods."""
    methods = []

    # SAQ (if wheel installed)
    try:
        from haag_vq.methods.search.saq_index import SaqIndex
        for bpd in args.bpd_values:
            methods.append(SaqIndex(bpd=bpd, K=args.K, nprobe=args.nprobe,
                                    num_threads=args.num_threads))
    except ImportError:
        print("WARNING: saq wheel not installed — skipping SaqIndex")

    # Faiss IVF-PQ baseline
    try:
        from haag_vq.methods.search.faiss_ivfpq_index import FaissIvfPqIndex
        methods.append(FaissIvfPqIndex(K=args.K, m=args.pq_m,
                                       nbits=args.pq_nbits, nprobe=args.nprobe))
    except ImportError:
        print("WARNING: faiss not installed — skipping FaissIvfPqIndex")

    # PQ flat + IVF wrappers
    try:
        from haag_vq.methods.base_quantizer import ProductQuantizer
        from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
        from haag_vq.methods.search.ivf_quantized_index import IvfQuantizedIndex
        methods.append(FlatQuantizedIndex(ProductQuantizer(M=args.pq_m, Ks=256)))
        methods.append(IvfQuantizedIndex(
            quantizer_factory=lambda: ProductQuantizer(M=args.pq_m, Ks=256),
            K=args.K, nprobe=args.nprobe
        ))
    except ImportError:
        pass

    return methods


def main():
    parser = argparse.ArgumentParser(description="VQ benchmark harness")
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/benchmark.csv'))
    parser.add_argument('--bpd-values', type=float, nargs='+', default=[1.0, 2.0, 4.0])
    parser.add_argument('--K', type=int, default=4096)
    parser.add_argument('--nprobe', type=int, default=200)
    parser.add_argument('--pq-m', type=int, default=16)
    parser.add_argument('--pq-nbits', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=8)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    X, Q, gt = load_dataset(args.data_dir)
    print(f"Dataset: X={X.shape}, Q={Q.shape}, gt={gt.shape}")

    methods = build_methods(args)
    print(f"Methods to benchmark: {[type(m).__name__ for m in methods]}")

    rows = []
    for idx in methods:
        print(f"\nFitting {type(idx).__name__}...")
        t0 = time.perf_counter()
        idx.fit(X)
        build_time = time.perf_counter() - t0
        print(f"  build_time={build_time:.1f}s")

        result = benchmark_index(idx, X, Q, gt, ks=(1, 10, 100))
        result['build_time_s'] = build_time
        rows.append(result)
        print(f"  recall@10={result.get('recall@10', 'N/A'):.3f}  "
              f"qps={result.get('qps', 0):.0f}  "
              f"memory_mb={result.get('memory_mb', 0):.1f}  "
              f"recon_mse={result.get('recon_mse')}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(df[['method', 'recall@10', 'recall@100', 'qps', 'memory_mb', 'recon_mse']].to_string())

    if not args.no_plot:
        pareto_plot(rows, x='memory_mb', y='recall@10')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify script runs without data (dry-run)**

```bash
python src/haag_vq/benchmarks/run_benchmarks.py --help
```
Expected: help text printed, no import errors.

---

### Task 19: Cleanup — delete `saq.py`, update `methods/__init__.py`

**Files:**
- Delete: `src/haag_vq/methods/saq.py`
- Modify: `src/haag_vq/methods/__init__.py`

- [ ] **Step 1: Delete `saq.py`**

```bash
git rm src/haag_vq/methods/saq.py
```

- [ ] **Step 2: Update `methods/__init__.py`**

Remove the line `from .saq import SAQ` (or equivalent import of the old pure-Python class). Add:

```python
from .search import SaqIndex, FlatQuantizedIndex, IvfQuantizedIndex, FaissIvfPqIndex
```

- [ ] **Step 3: Verify no remaining imports of the old `saq.py` class**

```bash
grep -r "from haag_vq.methods.saq" src/ tests/
grep -r "from .saq import" src/
```
Expected: no output (all references removed).

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: no FAILures from the deleted module; new tests pass.

---

### Task 20: Final commit and PR submission

- [ ] **Step 1: Commit all test bench changes**

```bash
git add src/haag_vq/methods/base_search_index.py \
        src/haag_vq/methods/search/ \
        src/haag_vq/benchmarks/search_bench.py \
        src/haag_vq/benchmarks/run_benchmarks.py \
        src/haag_vq/methods/__init__.py \
        tests/test_saq_index.py \
        tests/test_flat_quantized.py \
        tests/test_ivf_quantized.py \
        tests/test_faiss_ivfpq.py
git commit -m "feat: add BaseSearchIndex ABC, search wrappers, unified benchmark harness"
```

- [ ] **Step 2: Open PR to vector-quantization `main`**

```bash
git push origin feat/test-bench-search-index
gh pr create \
  --title "feat: BaseSearchIndex + search wrappers + unified benchmark harness" \
  --body "Adds BaseSearchIndex ABC, SaqIndex/FlatQuantizedIndex/IvfQuantizedIndex/FaissIvfPqIndex wrappers, search_bench.py harness, rewrites run_benchmarks.py. Deletes pure-Python saq.py placeholder. See design doc for full rationale." \
  --base main
```

- [ ] **Step 3: Verify all 4 SAQ wheels expose the expected API via CI**

On each branch (main, gpu, feat/optimal-codebook, gpu-codebook), confirm:
```bash
python -c "
import sys; sys.path.insert(0, 'python'); sys.path.insert(0, 'build_python/python/bindings/Release')
import saq
ivf = saq.IVF()
required = {'fit', 'decompress', 'search', 'search_batch', 'construct', 'set_codebooks', 'set_gaussian_codebooks', 'save', 'load'}
actual = set(m for m in dir(ivf) if not m.startswith('_'))
missing = required - actual
assert not missing, f'Missing methods: {missing}'
print('API surface OK')
"
```
Expected: `API surface OK` on all four branches.
