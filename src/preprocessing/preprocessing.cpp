// src/preprocessing/preprocessing.cpp
#include "saq/preprocessing/preprocessing.h"

namespace saq {

PreprocessingResult fit_ivf_preprocessing(
    const FloatRowMat& X,
    int   K,
    int   seed,
    bool  apply_pca)
{
    PreprocessingResult result;

    if (apply_pca) {
        PCAFit pca_fit;
        result.pca = pca_fit.fit(X);
        // Apply PCA rotation to training data before clustering.
        // FloatVec is a row vector (1, D), so no transpose before broadcast.
        FloatRowMat X_pca = (X.rowwise() - result.pca.mean) * result.pca.rotation;
        KMeans km(K, /*max_iter=*/25, seed);
        result.kmeans = km.fit(X_pca);
    } else {
        // Identity PCA result
        const int D = static_cast<int>(X.cols());
        result.pca.mean      = FloatVec::Zero(D);
        result.pca.rotation  = FloatRowMat::Identity(D, D);
        result.pca.variances = FloatVec::Ones(D);

        KMeans km(K, /*max_iter=*/25, seed);
        result.kmeans = km.fit(X);
    }

    return result;
}

} // namespace saq
