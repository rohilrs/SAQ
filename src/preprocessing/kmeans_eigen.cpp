// src/preprocessing/kmeans_eigen.cpp
// Compiled only when SAQ_USE_FAISS=OFF (Windows default).
// Lloyd's algorithm with k-means++ initialization.
#ifndef SAQ_USE_FAISS

#include "saq/preprocessing/kmeans.h"
#include <limits>
#include <random>
#include <vector>

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
