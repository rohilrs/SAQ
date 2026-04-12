#pragma once
// include/saq/preprocessing/pca.h

#include "saq/defines.h"

namespace saq {

struct PCAResult {
    FloatVec   mean;      // (D,)  — training mean
    FloatRowMat rotation; // (D,D) — orthogonal rotation matrix (columns = eigenvectors)
    FloatVec   variances; // (D,)  — per-dim variance in rotated space
};

/// Compute PCA from a data matrix.  Implementation lives in pca_eigen.cpp
/// or pca_faiss.cpp depending on SAQ_USE_FAISS compile flag.
class PCAFit {
public:
    /// Fit PCA to training data X (N, D). Returns rotation + mean + variances.
    PCAResult fit(const FloatRowMat& X) const;
};

} // namespace saq
