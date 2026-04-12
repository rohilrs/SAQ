#pragma once
// include/saq/preprocessing/kmeans.h

#include <cstdint>
#include <vector>

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
