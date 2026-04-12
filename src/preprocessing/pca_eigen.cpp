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

    // 1. Compute mean (FloatVec is a row vector (1, D))
    result.mean = X.colwise().mean();

    // 2. Center data (no transpose: mean is already a row vector)
    FloatRowMat Xc = X.rowwise() - result.mean;

    // 3. Covariance via SVD of centered data (economy mode)
    //    SVD(Xc) = U S Vt  =>  cov = Vt^T diag(S^2/(N-1)) Vt
    Eigen::BDCSVD<FloatRowMat> svd(Xc, Eigen::ComputeThinV);
    result.rotation = svd.matrixV(); // (D, D) — columns = eigenvectors
    // Singular values come back as a column vector; convert to our row-vector FloatVec
    auto sv = svd.singularValues();
    int Kmin = static_cast<int>(sv.size());
    result.variances.resize(1, D);
    for (int i = 0; i < Kmin;  ++i) result.variances[i] = sv[i] * sv[i] / (N - 1);
    for (int i = Kmin; i < D; ++i) result.variances[i] = 0.0f;

    return result;
}

} // namespace saq
#endif // !SAQ_USE_FAISS
