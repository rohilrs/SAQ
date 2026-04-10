#pragma once
// include/saq/preprocessing/preprocessing.h

#include "saq/preprocessing/pca.h"
#include "saq/preprocessing/kmeans.h"

namespace saq {

struct PreprocessingResult {
    PCAResult    pca;
    KMeansResult kmeans;
};

/// Run full IVF preprocessing: PCA (optional) then K-means on (possibly
/// rotated) data.  If apply_pca=false the PCAResult fields are identity
/// (zero mean, identity rotation, ones variances).
PreprocessingResult fit_ivf_preprocessing(
    const FloatRowMat& X,
    int   K,
    int   seed      = 0,
    bool  apply_pca = true
);

} // namespace saq
