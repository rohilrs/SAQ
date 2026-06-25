#pragma once

/// @file ivf_index.h
/// @brief Inverted File Index aligned with reference repository.
///
/// Ported from reference index/ivf.hpp.
/// Uses SaqCluData clusters, SaqData quantization plans, and
/// SAQSearcher for SIMD-accelerated search.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <fmt/core.h>

#include "saq/codebook_encoder.h"
#include "saq/cluster_data.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/initializer.h"
#include "saq/memory.h"
#include "saq/pool.h"
#include "saq/preprocessing/codebook_builder.h"
#include "saq/preprocessing/preprocessing.h"
#include "saq/quantization_plan.h"
#include "saq/quantizer.h"
#include "saq/saq_estimator.h"
#include "saq/saq_quantizer.h"
#include "saq/saq_searcher.h"
#include "saq/stopw.h"

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

namespace saq {

class IVF {
  public:
    QuantMetrics quant_metrics_;

  public:
    /// Row-major uint16 matrix used to cache raw quantization codes per
    /// (cluster, segment) during fit(), so decompress() can reconstruct
    /// approximate vectors without inverting the bit-packed SIMD layout.
    using RawCodeMat = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  protected:
    size_t num_data_;
    size_t num_dim_;
    size_t num_cen_;
    QuantizeConfig cfg_;
    std::unique_ptr<Initializer> initer_ = nullptr;
    std::vector<SaqCluData> parallel_clusters_;
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SaqDataMaker> saq_data_maker_;

    // Maps global vector ID -> {cluster_idx, local_idx_within_cluster}.
    // Populated by construct() only when raw_codes_out is non-null (the
    // fit() path). Used by decompress() to
    // locate a vector's slot inside parallel_clusters_.
    std::unordered_map<PID, std::pair<size_t, size_t>> id_to_location_;

    // PCA state stored during fit() for use in decompress() inverse rotation.
    bool        pca_applied_ = false;
    FloatVec    pca_mean_;       // (1, D) original-space mean (row vector)
    FloatRowMat pca_rotation_;   // (D, D) rotation applied during fit

    // Raw quantization codes cached during fit() for decompress().
    // Layout: raw_codes_[cluster_idx][segment_idx] is a
    //   (num_vectors_in_cluster, num_dim_padded_for_segment) uint16 matrix
    // where row k holds the integer codes for the k-th vector in that
    // cluster for that segment. Only populated when fit() was called; empty
    // when construct() was called directly (enterprise path, no decompress).
    std::vector<std::vector<RawCodeMat>> raw_codes_;

    // Codebook support
    bool has_codebooks_ = false;
    std::vector<std::vector<DimensionCodebook>> codebooks_;       // from set_codebooks()
    std::vector<std::vector<float>> gaussian_codebook_centroids_; // base[bits][entry]
    std::vector<float> residual_stds_;                            // per-dim std for gaussian path
    bool      derive_codebooks_ = false;  // derive natively from data in construct()
    bool      exact_codebooks_ = false;   // when deriving, use exact DP instead of Lloyd
    LloydOpts lloyd_opts_{};


    void allocate_clusters(const std::vector<size_t> &cluster_sizes);

    void prepare_initer(const FloatRowMat *centroids) {
        if (num_cen_ < 20000ul) {
            this->initer_ = std::make_unique<FlatInitializer>(num_dim_, num_cen_);
        } else {
            CHECK(false) << "HNSW not implemented\n";
        }
        if (centroids) {
            initer_->set_centroids(*centroids);
        }
    }

    void free_memory() {
        initer_.reset();
        parallel_clusters_.clear();
        saq_data_maker_.reset();
    }

  public:
    explicit IVF() = default;
    explicit IVF(size_t n, size_t num_dim, size_t k, QuantizeConfig cfg)
        : num_data_(n), num_dim_(num_dim), num_cen_(k), cfg_(std::move(cfg)),
          saq_data_maker_(std::make_unique<SaqDataMaker>(cfg_, num_dim)) {}
    IVF(const IVF &) = delete;
    ~IVF() { free_memory(); }

    auto num_data() const { return num_data_; }
    auto num_dim() const { return num_dim_; }
    auto &get_config() const { return cfg_; }
    auto get_initer() const { return initer_.get(); }
    const SaqData *get_saq_data() const { return saq_data_.get(); }
    auto &get_pclusters() const { return parallel_clusters_; }

    /// Build the IVF index from caller-supplied data, centroids, and
    /// per-vector cluster assignments. When @p raw_codes_out is non-null,
    /// this additionally fills it with per-cluster, per-segment raw uint16
    /// codes and populates id_to_location_ for decompress(). When null
    /// (default), neither side-effect happens -- the zero-overhead path.
    void construct(const FloatRowMat &data, const FloatRowMat &centroids,
                   const PID *cluster_ids, int num_threads = 64,
                   bool use_1_centroid = false,
                   std::vector<std::vector<RawCodeMat>> *raw_codes_out = nullptr);

    /// Self-contained preprocessing + construction from raw (N, D) vectors.
    /// Runs PCA (optional), k-means, and construct() internally. Also
    /// caches raw quantization codes so decompress() can reconstruct the
    /// vectors later.
    void fit(const FloatRowMat &X,
             bool apply_pca   = true,
             int  K           = 4096,
             int  seed        = 0,
             int  num_threads = 8);

    /// Approximate reconstruction from cached raw codes.
    /// @param ids global vector IDs (as returned by search()).
    /// @return (ids.size(), num_dim_) matrix in original (pre-PCA) space.
    ///
    /// The reconstructed vector's L2 norm is forced to equal the original
    /// vector's norm (via the cached per-vector o_l2norm factor); the
    /// direction is the CAQ-quantized direction. This differs from the
    /// internally-stored o_a reconstruction used at search time, but gives
    /// a usable pre-PCA-space approximation for downstream consumers.
    ///
    /// Requires fit() to have been called (not construct() alone); throws
    /// via CHECK if raw_codes_ is empty. Note that raw codes and PCA state
    /// are in-memory only -- a saved and re-loaded index does not support
    /// decompress() until fit() is re-run.
    FloatRowMat decompress(const std::vector<PID> &ids) const;

    void save(const char *filename) const;
    void load(const char *filename);

    template <DistType kDistType = DistType::Any>
    void search(const Eigen::RowVectorXf &ori_query,
                size_t topk, size_t nprobe, SearcherConfig searcher_cfg,
                PID *results, QueryRuntimeMetrics *runtime_metrics = nullptr);

    template <DistType kDistType = DistType::Any>
    void estimate(const Eigen::RowVectorXf &ori_query,
                  size_t nprobe, SearcherConfig searcher_cfg,
                  std::vector<std::pair<PID, float>> &dist_list, std::vector<float> *fast_dist_list = nullptr,
                  std::vector<float> *vars_dist_list = nullptr, QueryRuntimeMetrics *runtime_metrics = nullptr);

    size_t k() const { return num_cen_; }

    bool has_codebooks() const { return has_codebooks_; }

    /// Set explicit per-segment, per-dimension codebooks.
    void set_codebooks(std::vector<std::vector<DimensionCodebook>> cbs) {
        CHECK(!derive_codebooks_) << "set_codebooks: native derivation was already enabled via set_derive_codebooks(); these modes are mutually exclusive";
        codebooks_ = std::move(cbs);
        has_codebooks_ = true;
    }

    /// Enable native, data-driven codebook derivation during construct().
    /// Mutually exclusive with set_codebooks()/set_gaussian_codebooks() (those
    /// inject precomputed codebooks instead). After calling, construct() will
    /// build per-dim Lloyd codebooks from the data matrix at the bit-counts
    /// allocated by quant_plan.
    void set_derive_codebooks(LloydOpts opts = {}) {
        CHECK(codebooks_.empty()) << "set_derive_codebooks: explicit codebooks already set via set_codebooks(); these modes are mutually exclusive";
        CHECK(gaussian_codebook_centroids_.empty()) << "set_derive_codebooks: gaussian codebooks already set; these modes are mutually exclusive";
        lloyd_opts_ = opts;
        derive_codebooks_ = true;
        has_codebooks_ = true;
    }

    /// Like set_derive_codebooks() but derives EXACT (globally optimal) per-dim
    /// codebooks (build_all_dims_exact) instead of Lloyd. `max_bits` caps the
    /// largest bit-rate built. Exact, parameter-free, faster than Lloyd.
    void set_derive_codebooks_exact(size_t max_bits = 13) {
        CHECK(codebooks_.empty()) << "set_derive_codebooks_exact: explicit codebooks already set; mutually exclusive";
        CHECK(gaussian_codebook_centroids_.empty()) << "set_derive_codebooks_exact: gaussian codebooks already set; mutually exclusive";
        lloyd_opts_.max_bits = max_bits;
        derive_codebooks_ = true;
        exact_codebooks_ = true;
        has_codebooks_ = true;
    }

    /// Set Gaussian base codebooks + per-dimension variances.
    /// At construct time, each dim's codebook = base_codebook[bits] * std[dim].
    void set_gaussian_codebooks(
        std::vector<std::vector<float>> base_centroids,
        std::vector<float> residual_stds) {
        CHECK(!derive_codebooks_) << "set_gaussian_codebooks: native derivation was already enabled via set_derive_codebooks(); these modes are mutually exclusive";
        gaussian_codebook_centroids_ = std::move(base_centroids);
        residual_stds_ = std::move(residual_stds);
        has_codebooks_ = true;
    }

    void set_variance(FloatVec vars) {
        saq_data_maker_->set_variance(std::move(vars));
    }

    void printQPlan(const SaqData *data) {
        LOG(INFO) << "Dynamic bits allocation plan:";
        size_t dims_sum = 0;
        std::string log = fmt::format("{}bits: ", cfg_.avg_bits);
        dims_sum = 0;
        for (const auto &seg : data->quant_plan) {
            size_t dim_len = seg.first;
            size_t bits = seg.second;
            log += fmt::format("| {} -> {} ({}d {}b) ", dims_sum, dims_sum + dim_len, dim_len, bits);
            dims_sum += dim_len;
        }
        LOG(INFO) << log;
    }
};

// ============================================================================
// Template method implementations (must be in header)
// ============================================================================

template <DistType kDistType>
inline void IVF::search(const Eigen::RowVectorXf &ori_query, size_t topk, size_t nprobe,
                        SearcherConfig searcher_cfg, PID *results,
                        QueryRuntimeMetrics *runtime_metrics) {
    CHECK_EQ(ori_query.cols(), static_cast<Eigen::Index>(num_dim_));
    std::vector<Candidate> centroid_dist(nprobe);
    this->initer_->centroids_distances(ori_query, nprobe, searcher_cfg.dist_type, centroid_dist);
    ResultPool KNNs(topk, searcher_cfg.dist_type == DistType::IP);
    SAQSearcher<kDistType> searchers(*saq_data_.get(), searcher_cfg, ori_query);
    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        searchers.searchCluster(&parallel_clusters_[cid], KNNs);
    }
    KNNs.copy_results(results);
    if (runtime_metrics) {
        *runtime_metrics = searchers.getRuntimeMetrics();
    }
}

template <DistType kDistType>
inline void IVF::estimate(const Eigen::RowVectorXf &ori_query, size_t nprobe,
                          SearcherConfig searcher_cfg,
                          std::vector<std::pair<PID, float>> &dist_list, std::vector<float> *fast_dist_list,
                          std::vector<float> *vars_dist_list, QueryRuntimeMetrics *runtime_metrics) {
    CHECK_EQ(ori_query.cols(), static_cast<Eigen::Index>(num_dim_));
    std::vector<Candidate> centroid_dist(nprobe);
    this->initer_->centroids_distances(ori_query, nprobe, searcher_cfg.dist_type, centroid_dist);
    SaqCluEstimator<kDistType> estimator(*saq_data_.get(), searcher_cfg, ori_query);
    for (size_t j = 0; j < nprobe; ++j) {
        PID cid = centroid_dist[j].id;
        const auto &pcluster = parallel_clusters_[cid];
        estimator.prepare(&pcluster);
#if defined(__AVX512F__)
        float PORTABLE_ALIGN64 fastdist_t[KFastScanSize];
        float PORTABLE_ALIGN64 vardist_t[KFastScanSize];
        for (size_t vec_idx = 0; vec_idx < pcluster.num_vec_; ++vec_idx) {
            if (vec_idx % KFastScanSize == 0) {
                __m512 t[2];
                estimator.compFastDist(vec_idx / KFastScanSize, t);
                _mm512_store_ps(fastdist_t, t[0]);
                _mm512_store_ps(fastdist_t + 16, t[1]);
                estimator.varsEstDist(vec_idx / KFastScanSize, t);
                _mm512_store_ps(vardist_t, t[0]);
                _mm512_store_ps(vardist_t + 16, t[1]);
            }
            PID data_id = pcluster.ids()[vec_idx];
            float est_dist = estimator.compAccurateDist(vec_idx);
            dist_list.emplace_back(data_id, est_dist);
            if (fast_dist_list) {
                fast_dist_list->push_back(fastdist_t[vec_idx % KFastScanSize]);
            }
            if (vars_dist_list) {
                vars_dist_list->push_back(vardist_t[vec_idx % KFastScanSize]);
            }
        }
#else
        for (size_t vec_idx = 0; vec_idx < pcluster.num_vec_; ++vec_idx) {
            PID data_id = pcluster.ids()[vec_idx];
            float est_dist = estimator.compAccurateDist(vec_idx);
            dist_list.emplace_back(data_id, est_dist);
        }
#endif
    }
    if (runtime_metrics) {
        *runtime_metrics = estimator.getRuntimeMetrics();
    }
}

} // namespace saq
