// src/preprocessing/pca_faiss.cpp
// Compiled only when SAQ_USE_FAISS=ON (Linux default).
#ifdef SAQ_USE_FAISS

#include "saq/preprocessing/pca.h"

#include <cstring>

#include <faiss/VectorTransform.h>
#include <faiss/utils/distances.h>

namespace saq {

PCAResult PCAFit::fit(const FloatRowMat& X) const {
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());

    faiss::PCAMatrix pca(D, D, /*eigenvalue power*/ 0, /*random rotation*/ false);
    // Faiss expects row-major float* (which Eigen FloatRowMat is)
    pca.train(N, X.data());

    PCAResult result;

    // Mean (FloatVec is a row vector (1, D))
    result.mean.resize(1, D);
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
    result.variances.resize(1, D);
    std::memcpy(result.variances.data(), pca.eigenvalues.data(), D * sizeof(float));

    return result;
}

} // namespace saq
#endif // SAQ_USE_FAISS
