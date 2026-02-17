/// @file ivf_index.cpp
/// @brief Implementation of non-template IVF methods.

#include "index/ivf_index.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <vector>

namespace saq {

void IVF::allocate_clusters(const std::vector<size_t> &cluster_sizes) {
    parallel_clusters_.clear();
    parallel_clusters_.reserve(num_cen_);
    for (size_t i = 0; i < num_cen_; ++i) {
        parallel_clusters_.emplace_back(cluster_sizes[i], saq_data_->quant_plan, cfg_.use_compact_layout);
    }
    LOG(INFO) << "Initializing done... num_points: " << num_data_;
}

void IVF::construct(const FloatRowMat &data, const FloatRowMat &centroids, const PID *cluster_ids,
                    int num_threads, bool use_1_centroid) {
    LOG(INFO) << "Start IVF construction...\n";

    // 1. prepare initializer
    prepare_initer(&centroids);

    // 2. prepare SAQ data
    {
        if (!saq_data_maker_->is_variance_set()) {
            saq_data_maker_->compute_variance(data);
        }
        saq_data_ = saq_data_maker_->return_data();
        printQPlan(saq_data_.get());
    }

    // 3. prepare clusters
    std::vector<std::vector<PID>> id_lists(num_cen_);
    {
        std::vector<size_t> counts(num_cen_, 0);
        for (size_t i = 0; i < num_data_; ++i) {
            PID cid = cluster_ids[i];
            CHECK_LE(cid, static_cast<PID>(num_cen_)) << "Bad cluster id\n";
            id_lists[cid].push_back(static_cast<PID>(i));
            counts[cid] += 1;
        }
        allocate_clusters(counts);
    }

    // 4. quantize clusters
    {
        FloatVec tot_avg_centroid;
        if (use_1_centroid) {
            tot_avg_centroid = data.colwise().mean();
        }
        SAQuantizer saq_quantizer_(saq_data_.get());
        LOG(INFO) << "Starting quantization of " << num_cen_ << " clusters...";
        StopW stopw;
#ifdef SAQ_USE_OPENMP
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
#else
        (void)num_threads;
#endif
        for (size_t i = 0; i < num_cen_; ++i) {
            if (i % 500 == 0) {
                LOG(INFO) << "Quantizing cluster " << i << "/" << num_cen_
                          << " (size=" << id_lists[i].size() << ")";
            }
            const FloatVec &cur_centroid = use_1_centroid ? tot_avg_centroid : FloatVec(centroids.row(i));
            auto &clu = parallel_clusters_[i];
            saq_quantizer_.quantize_cluster(data, cur_centroid, id_lists[i], clu);
        }
        auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;
        LOG(INFO) << "Quantization done. tm: " << tm_ms / 1e3 << " S";
    }
}

void IVF::save(const char *filename) const {
    if (parallel_clusters_.empty()) {
        LOG(ERROR) << "IVF not constructed\n";
        return;
    }
    std::ofstream output(filename, std::ios::binary);
    output.write(reinterpret_cast<const char *>(&num_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&num_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&num_cen_), sizeof(size_t));
    this->initer_->save(output, filename);
    saq_data_->save(output);
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cen_);
    for (const auto &cur_cluster : parallel_clusters_) {
        cluster_sizes.push_back(cur_cluster.num_vec_);
    }
    output.write(reinterpret_cast<const char *>(cluster_sizes.data()), sizeof(size_t) * num_cen_);
    for (const auto &pclu : parallel_clusters_) {
        pclu.save(output);
    }
    output.close();
}

void IVF::load(const char *filename) {
    free_memory();
    LOG(INFO) << "Loading IVF...\n";
    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());
    input.read(reinterpret_cast<char *>(&this->num_data_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&this->num_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&this->num_cen_), sizeof(size_t));
    prepare_initer(nullptr);
    this->initer_->load(input, filename);
    saq_data_ = std::make_unique<SaqData>();
    saq_data_->load(input);
    std::vector<size_t> cluster_sizes(num_cen_, 0);
    input.read(reinterpret_cast<char *>(cluster_sizes.data()), sizeof(size_t) * num_cen_);
    DCHECK_EQ(num_data_, std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0)));
    allocate_clusters(cluster_sizes);
    for (size_t i = 0; i < num_cen_; ++i) {
        parallel_clusters_[i].load(input);
    }
    input.close();
    LOG(INFO) << "Index loaded\n";
}

} // namespace saq
