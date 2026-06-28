// src/preprocessing/pca_eigen.cpp
// Compiled only when SAQ_USE_FAISS=OFF.
// PCA via the (D x D) covariance matrix + symmetric eigensolver. For N >> D this
// is dramatically faster than BDCSVD on the full N x D matrix (which was
// ~O(N*D^2) with a huge constant and capped at ~200K vectors). The covariance
// cross-product is one GEMM (parallelized by Eigen/OpenMP); the eigendecomp is
// only O(D^3). Mathematically identical: eigenvectors of cov == right singular
// vectors of the centered data; eigenvalues == singular_value^2/(N-1).
#ifndef SAQ_USE_FAISS

#include "saq/preprocessing/pca.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace saq {

PCAResult PCAFit::fit(const FloatRowMat& X) const {
    PCAResult result;
    const int N = static_cast<int>(X.rows());
    const int D = static_cast<int>(X.cols());

    // 1. Mean (row vector 1 x D) and centered data.
    result.mean = X.colwise().mean();
    FloatRowMat Xc = X.rowwise() - result.mean;

    // 2. Covariance (D x D) via one float GEMM (same precision as the prior
    //    BDCSVD-on-float path), eigendecomposed in double for a stable solve.
    Eigen::MatrixXd cov =
        (Xc.transpose() * Xc).template cast<double>() / static_cast<double>(N - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    const Eigen::VectorXd& evals = es.eigenvalues();   // ascending
    const Eigen::MatrixXd& evecs = es.eigenvectors();  // columns = eigenvectors

    // 3. Emit rotation columns + variances ordered by DESCENDING variance, to
    //    match the previous BDCSVD (singular values descending) convention that
    //    downstream variance-ordered segmentation relies on.
    result.rotation.resize(D, D);
    result.variances.resize(1, D);
    for (int i = 0; i < D; ++i) {
        const int src = D - 1 - i;
        result.rotation.col(i) = evecs.col(src).cast<float>();
        const double v = evals[src];
        result.variances[i] = static_cast<float>(v > 0.0 ? v : 0.0);
    }
    return result;
}

} // namespace saq
#endif // !SAQ_USE_FAISS
