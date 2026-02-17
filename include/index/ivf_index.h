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
#include <vector>

#include <glog/logging.h>
#include <fmt/core.h>

#include "saq/cluster_data.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/initializer.h"
#include "saq/memory.h"
#include "saq/pool.h"
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

  protected:
    size_t num_data_;
    size_t num_dim_;
    size_t num_cen_;
    QuantizeConfig cfg_;
    std::unique_ptr<Initializer> initer_ = nullptr;
    std::vector<SaqCluData> parallel_clusters_;
    std::unique_ptr<SaqData> saq_data_;
    std::unique_ptr<SaqDataMaker> saq_data_maker_;

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

    void construct(const FloatRowMat &data, const FloatRowMat &centroids, const PID *cluster_ids,
                   int num_threads = 64, bool use_1_centroid = false);

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
