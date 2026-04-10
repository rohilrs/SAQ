# SAQ Python Packaging + Test Bench Integration — Architecture Design

## Overview

This project has two coupled goals:

1. **Package the SAQ C++ library as importable Python modules across four variants** — base CPU, GPU, Gaussian-codebook, and GPU+codebook (stub). Each variant lives on its own branch and builds a pybind11 module that exposes an identical Python API surface, so consuming code is variant-agnostic and the variant is selected by which wheel is installed.

2. **Unify the `vector-quantization/` test bench around a search-oriented benchmark interface.** The current `BaseQuantizer` track measures reconstruction MSE only — this is the wrong primary metric for fair comparison across VQ method families. Introduce `BaseSearchIndex` as the **primary** interface for all VQ methods. Every method (SAQ, PQ, OPQ, SQ, RaBitQ, Faiss baselines) gets wrapped to implement it. All methods are measured on the same axes: recall@k, QPS, memory footprint. Reconstruction MSE is retained as a secondary metric on the same benchmark output, not a separate track.

Both changes serve the same goal: a **fair, methodologically sound benchmark** of SAQ against competing VQ methods. Measuring SAQ by reconstruction MSE alone is unfair because CAQ optimizes cosine similarity, not MSE; measuring PQ only on MSE is similarly incomplete because it misses the search quality that matters for downstream tasks. The unified `BaseSearchIndex` approach — with MSE reported as a secondary column — gives a complete comparison across method families.

## Motivation

The current state has three problems:

1. **No self-contained Python entry point for SAQ.** The existing bindings require users to run Python preprocessing scripts (`pca.py`, `ivf.py`, `compute_gt.py`) to generate `.fvecs` files, then manually feed them to `IVF.construct()`. This is fine for the sample programs but painful for integration into other Python code or a benchmark harness.

2. **The test bench's `SAQ` class is a pure-Python reimplementation** of scalar quantization with segmented bit allocation. It shares the name with the C++ library but not the algorithm — no IVF, no CAQ adjustment, no SIMD search, no fastscan. Comparing it against PQ/OPQ gives misleading results about "SAQ performance."

3. **The `BaseQuantizer` interface is category-mismatched with search-oriented methods.** SAQ's value proposition is end-to-end search recall, not reconstruction accuracy. CAQ's per-vector adjustment actively worsens reconstruction MSE to preserve cosine similarity. Measuring SAQ alongside PQ on MSE alone gives PQ an unfair advantage — they're being measured on their own objective while SAQ is measured on someone else's. Unifying around search metrics (recall@k) puts all methods on equal footing.

## Goals

- Four SAQ wheels installable via `pip install` (or `cmake --build` for development), each exposing the same `saq` Python API
- Self-contained `IVF.fit(X)` API that handles preprocessing internally
- `IVF.decompress(ids)` for reconstruction + benchmark MSE reporting
- New `BaseSearchIndex` ABC in the test bench as the **primary benchmark interface** for all VQ methods
- Wrapper classes (`SaqIndex`, `FlatQuantizedIndex`, `IvfQuantizedIndex`, `FaissIvfPqIndex`) that adapt existing and new methods to `BaseSearchIndex`
- Unified benchmark harness reporting recall@k, QPS, memory, and reconstruction MSE on a single output per method
- Pareto curve visualization (recall vs memory, recall vs QPS)
- PR submitted to `vector-quantization` collaborators

## Non-Goals

- **GPU+codebook kernel integration.** The `gpu-codebook` branch is a **stub only**. The GPU encode kernel currently does uniform scalar quantization; rewriting it for codebook lookup is a major feature deferred until after this packaging work. The stub exposes the same API surface but raises `NotImplementedError` at runtime.
- **Full Faiss IndexBase API compatibility.** The `BaseSearchIndex` interface targets the **standard level** (fit, search, search_with_scores, memory_footprint, save, load). Incremental add/remove, range search, and parameter tuning are not in scope.
- **Removing `BaseQuantizer`.** Existing `BaseQuantizer` classes (scalar, product, optimized product, RaBitQ) stay as-is. They are no longer the primary benchmark interface, but they remain the implementation detail that `FlatQuantizedIndex` and `IvfQuantizedIndex` wrap internally. No existing quantizer code gets rewritten — only a thin wrapper layer is added on top.
- **Production-grade Eigen K-means.** The Windows Eigen fallback is a development convenience capped at ~200K vectors, not a production benchmark surface. All reported numbers come from Linux Faiss builds.

## Architecture

### Repo Layout

```
┌─────────────────────── SAQ/ (C++ library, 4 branches) ───────────────────────┐
│                                                                               │
│   include/saq/preprocessing/          ← NEW                                   │
│     ├─ preprocessing.h  (fit_ivf_preprocessing() orchestrator)                │
│     ├─ pca.h            (PCAFit class)                                        │
│     └─ kmeans.h         (KMeans class)                                        │
│                                                                               │
│   src/preprocessing/                  ← NEW                                   │
│     ├─ pca_eigen.cpp    (always compiled, Eigen BDCSVD)                       │
│     ├─ pca_faiss.cpp    (SAQ_USE_FAISS only, faiss::PCAMatrix)                │
│     ├─ kmeans_eigen.cpp (always compiled, Lloyd's + k-means++)                │
│     └─ kmeans_faiss.cpp (SAQ_USE_FAISS only, faiss::Kmeans)                   │
│                                                                               │
│   include/index/ivf_index.h          ← MODIFY                                 │
│     ├─ IVF::fit(X, apply_pca, K, seed)      ← NEW                             │
│     ├─ IVF::decompress(ids) -> FloatRowMat  ← NEW                             │
│     ├─ IVF::construct(...)                    (existing, stays)               │
│     └─ IVF::search(...)                       (existing, stays)               │
│                                                                               │
│   python/bindings/saq_bindings.cpp   ← MODIFY (identical across branches)     │
│     └─ .def("fit", ...)                                                       │
│     └─ .def("decompress", ...)                                                │
│     └─ .def("set_codebooks", ...)   (runtime error on base/gpu)               │
│     └─ .def("set_gaussian_codebooks", ...) (runtime error on base/gpu)        │
│                                                                               │
│   python/saq/__init__.py             ← no new exports (methods on IVF)        │
│                                                                               │
│   CMakeLists.txt                     ← MODIFY (SAQ_USE_FAISS option)          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ cmake --build, produces .pyd/.so
                                       ▼
┌────────────────────── vector-quantization/ (test bench) ─────────────────────┐
│                                                                               │
│   src/haag_vq/methods/                                                        │
│     ├─ base_quantizer.py           (existing, unchanged — now a utility)      │
│     ├─ base_search_index.py        ← NEW ABC (primary benchmark interface)    │
│     ├─ saq.py                      ← DELETE (pure Python placeholder)         │
│     └─ search/                     ← NEW subpackage                           │
│         ├─ __init__.py                                                        │
│         ├─ saq_index.py            (SaqIndex wraps saq.IVF / saq.GpuIVF)      │
│         ├─ flat_quantized_index.py (wraps any BaseQuantizer w/ brute-force)   │
│         ├─ ivf_quantized_index.py  (wraps any BaseQuantizer w/ IVF shell)     │
│         └─ faiss_ivfpq_index.py    (wraps faiss.IndexIVFPQ as baseline)       │
│                                                                               │
│   src/haag_vq/benchmarks/                                                     │
│     ├─ run_benchmarks.py           ← REWRITE (dispatch all methods via        │
│     │                                        BaseSearchIndex)                 │
│     └─ search_bench.py             ← NEW (recall@k, QPS, memory, MSE, Pareto) │
│                                                                               │
│   tests/                                                                      │
│     ├─ test_saq.py                 ← REWRITE (test SaqIndex contract)         │
│     ├─ test_flat_quantized.py      ← NEW (contract test for wrapper)          │
│     └─ test_ivf_quantized.py       ← NEW (contract test for wrapper)          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### SAQ C++ Changes

#### 1. Preprocessing module (`include/saq/preprocessing/`)

**`pca.h`**

```cpp
namespace saq {

struct PCAResult {
    FloatVec mean;          // (D,)
    FloatRowMat rotation;   // (D, D) — orthogonal
    FloatVec variances;     // (D,) — per-dim variance in rotated space
};

class PCAFit {
public:
    PCAResult fit(const FloatRowMat& X) const;
};

} // namespace saq
```

**`kmeans.h`**

```cpp
namespace saq {

struct KMeansResult {
    FloatRowMat centroids;          // (K, D)
    std::vector<PID> assignments;   // (N,)
};

class KMeans {
public:
    explicit KMeans(int K, int max_iter = 25, int seed = 0);
    KMeansResult fit(const FloatRowMat& X) const;

private:
    int K_, max_iter_, seed_;
};

} // namespace saq
```

**`preprocessing.h`** (umbrella)

```cpp
namespace saq {

struct PreprocessingResult {
    PCAResult pca;
    KMeansResult kmeans;
};

PreprocessingResult fit_ivf_preprocessing(
    const FloatRowMat& X,
    int K,
    int seed = 0,
    bool apply_pca = true
);

} // namespace saq
```

#### 2. Faiss/Eigen dispatch pattern

Compile-time Strategy at link-time. No Faiss types leak into headers. Exactly one of `pca_eigen.cpp` / `pca_faiss.cpp` is compiled based on `SAQ_USE_FAISS`.

```cmake
# CMakeLists.txt (root)
if(WIN32)
    set(SAQ_USE_FAISS_DEFAULT OFF)
else()
    set(SAQ_USE_FAISS_DEFAULT ON)
endif()
option(SAQ_USE_FAISS "Use Faiss for PCA and k-means (Linux default ON)"
       ${SAQ_USE_FAISS_DEFAULT})

if(SAQ_USE_FAISS)
    find_package(faiss REQUIRED)
    target_link_libraries(saq PRIVATE faiss)
    target_compile_definitions(saq PRIVATE SAQ_USE_FAISS)
    target_sources(saq PRIVATE
        src/preprocessing/pca_faiss.cpp
        src/preprocessing/kmeans_faiss.cpp
    )
else()
    target_sources(saq PRIVATE
        src/preprocessing/pca_eigen.cpp
        src/preprocessing/kmeans_eigen.cpp
    )
endif()
```

On Linux, Faiss is expected as a system package (`apt install libfaiss-dev` or `conda install faiss-cpu` before build). FetchContent is rejected — Faiss's CMake is not FetchContent-friendly.

On Windows, `SAQ_USE_FAISS=OFF` by default. The Eigen fallback is **minimal** (~200 LOC total for PCA + K-means), single-threaded, documented as dev-only. Windows benchmarks are not reported.

#### 3. `IVF::fit()` and `IVF::decompress()`

```cpp
// include/index/ivf_index.h additions

class IVF {
public:
    // existing methods...

    /// Self-contained preprocessing + construction from raw vectors.
    /// Calls fit_ivf_preprocessing() then construct() internally.
    void fit(
        const FloatRowMat& X,
        bool apply_pca = true,
        int K = 4096,
        int seed = 0,
        int num_threads = 8
    );

    /// Reconstruct approximate vectors from stored codes.
    /// Returns (ids.size(), num_dim) matrix in original scale.
    FloatRowMat decompress(const std::vector<PID>& ids) const;
};
```

**`fit()` implementation sketch:**

```cpp
void IVF::fit(const FloatRowMat& X, bool apply_pca, int K, int seed, int num_threads) {
    PreprocessingResult pp = fit_ivf_preprocessing(X, K, seed, apply_pca);

    // Apply PCA to data and centroids if requested
    FloatRowMat X_proc = apply_pca ? (X.rowwise() - pp.pca.mean.transpose()) * pp.pca.rotation : X;
    FloatRowMat centroids_proc = apply_pca
        ? (pp.kmeans.centroids.rowwise() - pp.pca.mean.transpose()) * pp.pca.rotation
        : pp.kmeans.centroids;

    set_variance(pp.pca.variances);
    construct(X_proc, centroids_proc, pp.kmeans.assignments.data(), num_threads);
}
```

**`decompress()` implementation sketch** — based on verified CAQ invertibility:

```cpp
FloatRowMat IVF::decompress(const std::vector<PID>& ids) const {
    FloatRowMat result(ids.size(), num_dim_);
    for (size_t i = 0; i < ids.size(); ++i) {
        PID vid = ids[i];
        // Locate cluster containing vid (lookup table built once during fit/construct)
        auto [cluster_idx, local_idx] = find_vector(vid);
        const auto& cluster = parallel_clusters_[cluster_idx];

        // Reconstruct per segment:
        FloatVec o_rotated = FloatVec::Zero(num_dim_padded_);
        size_t offset = 0;
        for (size_t seg = 0; seg < saq_data_->base_datas.size(); ++seg) {
            const auto& seg_data = saq_data_->base_datas[seg];
            const auto& clu_seg = cluster.get_segment(seg);
            // Unpack codes (short MSB + long bits) for vector local_idx
            Eigen::VectorXi code = unpack_vector_code(clu_seg, local_idx, seg_data.num_bits);
            // Dequantize in normalized scale
            FloatVec o_a_norm = dequantize(code, seg_data);  // (code+0.5)*delta + v_mi
            // Retrieve o_l2norm for this vector from short factors
            float o_l2norm = clu_seg.factor_o_l2norm(local_idx / KFastScanSize)[local_idx % KFastScanSize];
            // Scale direction to match original norm
            float norm_a = o_a_norm.norm();
            if (norm_a > 0) o_a_norm *= (o_l2norm / norm_a);
            // Add rotated centroid, inverse-rotate
            FloatVec seg_rotated = o_a_norm + clu_seg.centroid();
            FloatVec seg_unrotated = seg_data.rotator
                ? seg_rotated.transpose() * seg_data.rotator->get_P().transpose()
                : seg_rotated;
            o_rotated.segment(offset, seg_data.num_dim_pad) = seg_unrotated;
            offset += seg_data.num_dim_pad;
        }
        // Inverse PCA rotation (if applied during fit)
        result.row(i) = apply_pca_inverse(o_rotated);
    }
    return result;
}
```

**Decompress semantics:** the reconstruction direction is the stored quantized code, the magnitude is matched to `o_l2norm` (the original vector's norm, stored per-vector in short factors). This is the fairest reconstruction SAQ is capable of given CAQ's objective (preserve direction + magnitude).

#### 4. Pybind11 bindings — identical across all 4 branches

```cpp
// python/bindings/saq_bindings.cpp — additions to IVF class

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
     "Self-contained preprocessing + construction.")

.def("decompress",
     [](const IVF &self, py::array_t<uint32_t> ids) -> Eigen::MatrixXf {
         auto buf = ids.request();
         const PID *ptr = static_cast<const PID *>(buf.ptr);
         std::vector<PID> id_vec(ptr, ptr + buf.size);
         py::gil_scoped_release release;
         return self.decompress(id_vec);
     },
     py::arg("ids"),
     "Reconstruct approximate vectors. Returns (N, D) float32 array.")

// Codebook methods — present on all branches, runtime-guarded:
.def("set_codebooks",
     [](IVF &self, Eigen::Ref<const FloatRowMat> codebooks) {
#ifdef SAQ_ENABLE_CODEBOOK
         self.set_codebooks(codebooks);
#else
         throw std::runtime_error("SAQ build does not support set_codebooks");
#endif
     },
     py::arg("codebooks"))

.def("set_gaussian_codebooks",
     [](IVF &self, Eigen::Ref<const FloatRowMat> base, Eigen::Ref<const FloatVec> stds) {
#ifdef SAQ_ENABLE_CODEBOOK
         self.set_gaussian_codebooks(base, stds);
#else
         throw std::runtime_error("SAQ build does not support set_gaussian_codebooks");
#endif
     },
     py::arg("base_codebook"), py::arg("residual_stds"))
```

The `SAQ_ENABLE_CODEBOOK` macro is defined only on the codebook branches, via CMake. This keeps the binding source **identical across branches** — only the compile-time flag differs.

GPU bindings in `saq_gpu_bindings.cpp` mirror the same `fit` and `decompress` signatures for `GpuIVF`. The `gpu-codebook` branch raises `NotImplementedError` on `set_gaussian_codebooks` — the GPU kernels don't support codebook lookup yet.

### Test Bench Changes

#### 1. `base_search_index.py`

The **primary benchmark interface** for all VQ methods. Every concrete class wraps either a native search index (SAQ, Faiss) or a `BaseQuantizer` with a search harness.

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Tuple
import numpy as np

class BaseSearchIndex(ABC):
    """Primary benchmark interface for VQ methods.

    All VQ methods (SAQ, PQ, OPQ, SQ, RaBitQ, Faiss baselines) implement
    this interface — either natively (SAQ, Faiss) or via wrapper classes
    that adapt BaseQuantizer implementations (FlatQuantizedIndex,
    IvfQuantizedIndex).

    Primary metrics: recall@k, QPS, memory_footprint.
    Secondary metric: reconstruction MSE (optional, via reconstruction_mse()).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        """Learn index from training vectors (N, D). Preprocessing is internal."""

    @abstractmethod
    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        """Return (nq, k) array of neighbor IDs."""

    @abstractmethod
    def search_with_scores(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) where both are (nq, k)."""

    @abstractmethod
    def memory_footprint(self) -> int:
        """Estimated index memory in bytes (for compression ratio computation)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore index from disk."""

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Return mean reconstruction MSE over a sample (optional metric).

        Default: returns None (method does not support reconstruction).
        Concrete classes override when their underlying quantizer/index
        supports decompress. Used as a secondary benchmark column.

        Args:
            X: Original vectors (N, D) used as ground truth.
            sample_ids: If provided, compute MSE over this subset.
                        If None, use all vectors.
        """
        return None
```

#### 2. `search/saq_index.py`

Single `SaqIndex` class with capability detection — adapts to whichever `saq` wheel is installed. No subclasses. The user chooses the variant by installing the corresponding wheel (`saq-cpu`, `saq-gpu`, `saq-codebook`, or `saq-gpu-codebook`).

```python
class SaqIndex(BaseSearchIndex):
    def __init__(
        self,
        bpd: float = 4.0,
        K: int = 4096,
        nprobe: int = 200,
        use_gpu: bool = False,
        use_codebook: bool = False,
        num_threads: int = 8,
    ) -> None:
        import saq
        self._saq = saq
        self._bpd, self._K, self._nprobe = bpd, K, nprobe
        self._use_gpu = use_gpu and hasattr(saq, 'GpuIVF')
        self._use_codebook = use_codebook
        self._num_threads = num_threads
        self._index = None
        self._metric: Literal['l2', 'ip'] = 'l2'

    def fit(self, X, metric='l2'):
        X = np.asarray(X, dtype=np.float32)
        self._metric = metric
        cfg = self._make_config(metric)
        IndexClass = self._saq.GpuIVF if self._use_gpu else self._saq.IVF
        self._index = IndexClass(X.shape[0], X.shape[1], self._K, cfg)
        if self._use_codebook:
            # Load precomputed codebooks from disk or compute from data
            self._setup_codebooks(X)
        self._index.fit(X, apply_pca=True, K=self._K, num_threads=self._num_threads)

    def search(self, Q, k):
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(self, Q, k):
        Q = np.asarray(Q, dtype=np.float32)
        searcher_cfg = self._make_searcher_config()
        return self._index.search_batch(Q, k, self._nprobe, searcher_cfg)

    def memory_footprint(self) -> int:
        # Approximate from stored codes + factors + centroids
        # Formula documented in SAQ C++ library
        ...

    def save(self, path): self._index.save(str(path))
    def load(self, path):
        self._index = self._saq.IVF()
        self._index.load(str(path))

    def reconstruction_mse(self, X, sample_ids=None):
        if sample_ids is None:
            sample_ids = np.arange(len(X), dtype=np.uint32)
        X_hat = self._index.decompress(sample_ids.astype(np.uint32))
        return float(np.mean((X[sample_ids] - X_hat) ** 2))
```

**Key decision:** `SaqIndex` inherits **only** from `BaseSearchIndex`. `BaseQuantizer` is no longer a benchmark interface. Reconstruction MSE is supported via the optional `reconstruction_mse()` method, which calls `IVF.decompress()` internally.

#### 3. `search/flat_quantized_index.py` — wraps any BaseQuantizer with brute-force search

```python
class FlatQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer with brute-force search.

    Fit: compresses all training vectors and stores codes.
    Search: decompresses stored codes, computes distance to query, top-k.

    Fair but slow — O(N) per query. Use for small/medium datasets or as
    a baseline that factors out IVF overhead.
    """

    def __init__(self, quantizer: BaseQuantizer):
        self._quantizer = quantizer
        self._codes: Optional[np.ndarray] = None
        self._X_ref: Optional[np.ndarray] = None  # kept for memory footprint

    def fit(self, X, metric='l2'):
        self._metric = metric
        self._quantizer.fit(X)
        self._codes = self._quantizer.compress(X)
        self._N, self._D = X.shape

    def search(self, Q, k):
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(self, Q, k):
        # Decompress full database, compute distances, top-k per query
        X_hat = self._quantizer.decompress(self._codes)
        if self._metric == 'l2':
            dists = cdist(Q, X_hat, metric='sqeuclidean')
            top_k = np.argpartition(dists, k, axis=1)[:, :k]
        else:  # 'ip'
            sims = Q @ X_hat.T
            top_k = np.argpartition(-sims, k, axis=1)[:, :k]
            dists = sims
        return top_k.astype(np.uint32), np.take_along_axis(dists, top_k, axis=1)

    def memory_footprint(self):
        return self._codes.nbytes

    def reconstruction_mse(self, X, sample_ids=None):
        if sample_ids is None:
            sample_ids = np.arange(len(X))
        X_hat = self._quantizer.decompress(self._codes[sample_ids])
        return float(np.mean((X[sample_ids] - X_hat) ** 2))

    def save(self, path): ...  # pickle quantizer + codes
    def load(self, path): ...
```

#### 4. `search/ivf_quantized_index.py` — wraps any BaseQuantizer with an IVF shell

```python
class IvfQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer inside a k-means IVF shell.

    Fit: runs k-means (K clusters), compresses residuals per cluster.
    Search: finds nearest nprobe centroids, searches within those clusters.

    Fair comparison with SAQ — both use IVF + per-cluster quantization.
    Uses faiss.Kmeans for clustering (already a dependency).
    """

    def __init__(
        self,
        quantizer_factory: Callable[[], BaseQuantizer],
        K: int = 4096,
        nprobe: int = 200,
    ):
        self._quantizer_factory = quantizer_factory
        self._K = K
        self._nprobe = nprobe
        self._centroids: Optional[np.ndarray] = None
        self._cluster_quantizers: list[BaseQuantizer] = []
        self._cluster_codes: list[np.ndarray] = []
        self._cluster_ids: list[np.ndarray] = []  # original vector IDs per cluster

    def fit(self, X, metric='l2'):
        import faiss
        self._metric = metric
        self._N, self._D = X.shape
        kmeans = faiss.Kmeans(self._D, self._K, seed=0)
        kmeans.train(X)
        self._centroids = kmeans.centroids
        assignments = kmeans.index.search(X, 1)[1].ravel()
        for c in range(self._K):
            mask = assignments == c
            if not mask.any():
                self._cluster_quantizers.append(None)
                self._cluster_codes.append(np.array([]))
                self._cluster_ids.append(np.array([], dtype=np.uint32))
                continue
            residuals = X[mask] - self._centroids[c]
            q = self._quantizer_factory()
            q.fit(residuals)
            self._cluster_quantizers.append(q)
            self._cluster_codes.append(q.compress(residuals))
            self._cluster_ids.append(np.where(mask)[0].astype(np.uint32))

    def search(self, Q, k): ...  # standard IVF probe + decompress + top-k
    def search_with_scores(self, Q, k): ...
    def memory_footprint(self): ...  # centroids + all cluster codes
    def reconstruction_mse(self, X, sample_ids=None): ...  # per-cluster decompress
    def save(self, path): ...
    def load(self, path): ...
```

#### 5. `search/faiss_ivfpq_index.py` — Faiss IVF-PQ baseline

```python
class FaissIvfPqIndex(BaseSearchIndex):
    """External baseline: faiss.IndexIVFPQ.

    Used as a reference point — if SAQ doesn't beat IVFPQ on recall@k
    at the same bpd, the algorithm isn't pulling its weight.
    """

    def __init__(self, K=4096, m=16, nbits=8, nprobe=200):
        import faiss
        self._faiss = faiss
        self._K, self._m, self._nbits, self._nprobe = K, m, nbits, nprobe
        self._index = None

    def fit(self, X, metric='l2'):
        quantizer = self._faiss.IndexFlatL2(X.shape[1])
        self._index = self._faiss.IndexIVFPQ(
            quantizer, X.shape[1], self._K, self._m, self._nbits
        )
        self._index.train(X)
        self._index.add(X)
        self._index.nprobe = self._nprobe

    def search(self, Q, k):
        _, ids = self._index.search(Q, k)
        return ids.astype(np.uint32)

    def search_with_scores(self, Q, k):
        dists, ids = self._index.search(Q, k)
        return ids.astype(np.uint32), dists

    def memory_footprint(self):
        # Approximation: K * D * 4 (centroids) + N * m bytes (PQ codes)
        ...

    def reconstruction_mse(self, X, sample_ids=None):
        # Faiss doesn't expose reconstruct() cheaply for IndexIVFPQ — skip
        return None

    def save(self, path): self._faiss.write_index(self._index, str(path))
    def load(self, path): self._index = self._faiss.read_index(str(path))
```

#### 6. `benchmarks/search_bench.py` — unified benchmark harness

```python
def benchmark_index(
    index: BaseSearchIndex,
    X: np.ndarray,
    Q: np.ndarray,
    ground_truth: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 100),
    mse_sample: int = 1000,
    repeats: int = 3,
) -> dict:
    """Benchmark a BaseSearchIndex implementation on recall, QPS, memory, MSE.

    Returns dict with:
        build_time_s: index construction time (already called before this fn)
        qps: queries per second, averaged over `repeats`
        memory_bytes: index memory footprint
        recall@1, recall@10, recall@100: recall metrics vs ground truth
        recon_mse: reconstruction MSE over a sample of `mse_sample` vectors
                   (None if index doesn't support reconstruction)
        method: class name of the index
        params: all __init__ kwargs of the index
    """
    results = {'method': type(index).__name__}
    # Search timing
    ...
    for k in ks:
        results[f'recall@{k}'] = compute_recall(ids, ground_truth, k)
    results['qps'] = ...
    results['memory_bytes'] = index.memory_footprint()

    # Secondary metric: reconstruction MSE (optional)
    if len(X) > mse_sample:
        sample_ids = np.random.default_rng(0).choice(len(X), mse_sample, replace=False)
    else:
        sample_ids = np.arange(len(X))
    mse = index.reconstruction_mse(X, sample_ids=sample_ids)
    results['recon_mse'] = mse  # may be None

    return results


def sweep_bpd(
    IndexClass: type,
    bpd_values: list[float],
    X: np.ndarray, Q: np.ndarray, ground_truth: np.ndarray,
    **index_kwargs,
) -> list[dict]:
    """Sweep bits-per-dim, return list for Pareto plotting."""


def compare_methods(
    indices: list[BaseSearchIndex],
    X: np.ndarray, Q: np.ndarray, ground_truth: np.ndarray,
) -> pd.DataFrame:
    """Run all methods, return a DataFrame comparing recall@k, QPS, memory, MSE."""


def pareto_plot(results: list[dict], x='memory_bytes', y='recall@10') -> None:
    """Plot Pareto curve. Supports x='memory_bytes'|'qps', y='recall@k'|'recon_mse'."""
```

Example comparison output:

| method | bpd | recall@10 | recall@100 | qps | memory_mb | recon_mse |
|---|---|---|---|---|---|---|
| FlatQuantizedIndex (PQ) | 2.0 | 78.3% | 68.2% | 120 | 24 | 0.0021 |
| IvfQuantizedIndex (PQ) | 2.0 | 89.5% | 84.8% | 3200 | 26 | 0.0021 |
| IvfQuantizedIndex (OPQ) | 2.0 | 91.8% | 87.3% | 3100 | 26 | 0.0015 |
| IvfQuantizedIndex (RaBitQ) | 2.0 | 93.1% | 90.5% | 2800 | 25 | 0.0018 |
| SaqIndex | 2.0 | 92.6% | 89.9% | 28 | 24 | 0.00008 |
| FaissIvfPqIndex | 2.0 | 90.4% | 85.6% | 3400 | 27 | — |

All methods reported on the same axes. SAQ's significantly lower MSE (consistent with CAQ's cosine-similarity optimization) is visible alongside its competitive recall. The slow QPS of SAQ (3-stage SIMD search has higher per-query overhead than IVFPQ's LUT lookup) is visible as a trade-off worth investigating.

### Data Flow

```
User code                 BaseSearchIndex             SaqIndex wrapper           saq.IVF (C++)
─────────                 ───────────────             ────────────────           ─────────────
idx = SaqIndex(bpd=2)
idx.fit(X)         ─────► fit(X, metric)       ─────► cfg = make_config()
                                                        _index = IVF(N,D,K,cfg)
                                                        _index.fit(X, pca=True) ──► PCAFit.fit
                                                                                    KMeans.fit
                                                                                    set_variance
                                                                                    construct(...)
                                                                                    ◄── done
idx.search(Q, 10)  ─────► search(Q, k)         ─────► _index.search_batch ────────► FlatInitializer
                                                                                    SAQSearcher
                                                                                    3-stage search
                                                                                    ◄── ids, dists

benchmark_index(idx, X, Q, gt)
  ├─ build_time  (measured in fit())
  ├─ search      (measured in search())
  ├─ recall@k    (compared against gt)
  ├─ qps         (N queries / wall time)
  └─ memory      (from index.memory_footprint())
```

### Integration Points

1. **Four SAQ wheels** built by branching and running `cmake --build build --target _saq_core`. Each wheel exposes the same API; the installed wheel determines the capabilities. Wheels are named:
   - `saq-cpu` (from `main`)
   - `saq-gpu` (from `gpu`)
   - `saq-codebook` (from `feat/optimal-codebook`)
   - `saq-gpu-codebook` (from `gpu-codebook`, stub)

2. **Test bench imports `saq` unconditionally.** Variant selection happens at wheel install time, not in Python. This keeps the test bench agnostic — users install the wheel that matches their hardware and experiment.

3. **Single unified benchmark track via `BaseSearchIndex`:**
   - `SaqIndex` wraps the C++ SAQ library
   - `FlatQuantizedIndex` wraps existing PQ/OPQ/SQ/RaBitQ with brute-force search
   - `IvfQuantizedIndex` wraps existing PQ/OPQ/SQ/RaBitQ with a k-means IVF shell (fair comparison with SAQ)
   - `FaissIvfPqIndex` is the external baseline
   - All methods report recall@k, QPS, memory, and (optionally) reconstruction MSE on the same output rows
   - `BaseQuantizer` classes are no longer called directly by the benchmark — they live inside the wrapper classes

4. **Ground truth computation** uses `faiss.IndexFlatL2` / `IndexFlatIP` on the original (un-quantized) data. Already present in test bench's preprocessing utilities.

## Risks & Trade-offs

| Risk | Severity | Mitigation |
|---|---|---|
| CAQ decompress correctness | Medium | Verified invertibility; semantic = "accurate-stage representation with original L2 norm"; validate against search path distances in tests |
| Faiss on Windows build | Low | Don't try. `SAQ_USE_FAISS=OFF` default on Windows, Eigen fallback is dev-only. Report Linux numbers only. |
| Eigen K-means at enterprise scale | Medium | Not intended for enterprise. Linux+Faiss is the production path. Document Windows cap at 200K vectors. |
| API drift across 4 branches | Medium | Shared `saq_bindings.cpp` source across branches; compile-time flags (`SAQ_ENABLE_CODEBOOK`) differentiate behavior; CI diff check on binding signatures |
| Test bench PR churn | Low-Med | Staged commits: (1) ABC, (2) wrapper, (3) bench, (4) cleanup. Submit as one reviewable PR. |
| GPU+codebook stub raising at runtime | Low | Clear error message, documented in README. Only raises if user tries `set_gaussian_codebooks` on GPU wheel. |
| PACE build reproducibility (CUDA + Faiss + AVX-512) | Medium | Dockerfile checked into repo; pin Faiss version and CUDA version; test build on PACE before major benchmarks |
| Binding GIL release correctness | Low | `py::gil_scoped_release` on long-running methods (fit, construct, search_batch); verified thread-safe in existing GPU branch |

## Show-Stoppers Resolved

Before starting implementation, two show-stoppers from the solution-architect review were verified:

1. **CAQ invertibility.** Confirmed: codes are stored post-adjustment, delta/v_mi are in normalized scale (post `rescale_vmx_to1`), `o_l2norm` is stored per-vector. Reconstruction is possible with ~60-80 LOC. Decompress semantic: direction-quantized vector scaled to match original L2 norm.

2. **Enterprise `fit()` feasibility.** Linux+Faiss path handles 10M × 1536D × K=4096 in minutes. Windows+Eigen is explicitly dev-only. The existing low-level `construct(X, centroids, cluster_ids)` API stays as an enterprise escape hatch for users who bring their own preprocessed data.

## Open Questions

None blocking. A few items to confirm with collaborators:

- **Test bench PR strategy:** one large PR or four staged PRs? Code-architect recommends four. Confirm with collaborators' review bandwidth.
- **Faiss version pinning:** use latest stable (1.7.4 as of writing) or match whatever PACE has installed?
- **Naming of four wheels:** `saq-cpu` vs `saq-base` vs `saq`? Match existing conventions in your research group.

## Branch Strategy

| Branch | Origin | Purpose | New files |
|---|---|---|---|
| `feat/ivf-fit-api` | `main` | Add `fit()`, `decompress()`, preprocessing module, `SAQ_USE_FAISS` | Preprocessing module, bindings, tests |
| `gpu-codebook` | `gpu` | Stub for GPU+codebook variant | Merge codebook bindings with stub GPU implementations |
| `feat/test-bench-search-index` | test bench `main` | `BaseSearchIndex` ABC + `SaqIndex` + search benchmark | All files in `vector-quantization/` changes |

The `feat/ivf-fit-api` branch is the **primary work branch** for the SAQ changes. It gets merged into `main` first, then cherry-picked/merged into `gpu`, `feat/optimal-codebook`, and `gpu-codebook` so all four branches share the new API.

## Implementation Order

**Phase 1: SAQ C++ library changes**
1. **SAQ: preprocessing module + `fit()` + `decompress()` + `SAQ_USE_FAISS` CMake** (on `feat/ivf-fit-api` branch from `main`)
2. **SAQ: pybind11 binding updates** (same branch)
3. **Verify on Linux with Faiss, Windows with Eigen fallback**
4. **Merge `feat/ivf-fit-api` → `main`**
5. **Propagate to `gpu`, `feat/optimal-codebook`** via merge
6. **Create `gpu-codebook` branch** from `gpu`, cherry-pick codebook bindings with stub implementations
7. **Build all 4 wheels, verify API identical across branches**

**Phase 2: Test bench core (BaseSearchIndex + SaqIndex)**
8. **Test bench: `BaseSearchIndex` ABC** (on new branch in `vector-quantization/`)
9. **Test bench: `SaqIndex` wrapper** (including `reconstruction_mse` override)
10. **Test bench: Contract tests for `SaqIndex`**

**Phase 3: Test bench wrappers for existing quantizers**
11. **Test bench: `FlatQuantizedIndex`** (generic wrapper over any `BaseQuantizer`)
12. **Test bench: `IvfQuantizedIndex`** (k-means IVF shell + per-cluster quantization)
13. **Test bench: `FaissIvfPqIndex`** (external baseline)
14. **Test bench: Contract tests for all wrappers**

**Phase 4: Benchmark harness + cleanup**
15. **Test bench: `search_bench.py`** (`benchmark_index`, `sweep_bpd`, `compare_methods`, `pareto_plot`)
16. **Test bench: Rewrite `run_benchmarks.py`** to dispatch all methods through `BaseSearchIndex`
17. **Test bench: Delete `saq.py`, update `methods/__init__.py`, remove SAQ import from `performance.py`**
18. **Test bench: Rewrite `tests/test_saq.py` against `BaseSearchIndex` contract**
19. **Submit test bench PR** to collaborators

Phase 1 is 4 SAQ PRs (one per branch, feat branch merged first). Phase 2-4 is one large PR to `vector-quantization` (can be split into 2-3 commits for reviewability: ABC+SaqIndex, then wrappers, then harness+cleanup).
