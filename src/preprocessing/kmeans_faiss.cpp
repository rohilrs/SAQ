// src/preprocessing/kmeans_faiss.cpp
// Compiled only when SAQ_USE_FAISS=ON (Linux default).
#ifdef SAQ_USE_FAISS

#include "saq/preprocessing/kmeans.h"

#include <cstring>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

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
