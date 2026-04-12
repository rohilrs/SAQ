// tests/test_fit_decompress.cpp
//
// Plain C++ smoke test for saq::IVF::fit() and saq::IVF::decompress().
// No gtest dependency — uses assert() + a hand-rolled main(). Each scenario
// prints "<name>: OK" on success; main() exits 0 if all pass.
//
// Scenarios:
//   1. FitSmoke                         — fit() does not throw; dims are right.
//   2. DecompressShapeAndFinite         — decompress() returns correct shape,
//                                         all values finite.
//   3. DecompressMSEReasonable_NoPCA    — reconstruction MSE < 0.2 at 4 bpd.
//   4. DecompressMSEReasonable_WithPCA  — same with PCA enabled.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "index/ivf_index.h"
#include "saq/config.h"
#include "saq/defines.h"

using saq::FloatRowMat;
using saq::IVF;
using saq::PID;
using saq::QuantizeConfig;

// Row-wise L2-normalized Gaussian matrix. Unit-norm rows are the standard
// setting for SAQ benchmarks (dbpedia-style embeddings).
static FloatRowMat make_random(int N, int D, int seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    FloatRowMat X(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            X(i, j) = nd(rng);
    // L2-normalize rows. colwise()/rowwise().norm() divides each row by its
    // norm (broadcasts the column vector of norms across columns).
    X = X.array().colwise() / X.rowwise().norm().array();
    return X;
}

static void test_fit_smoke() {
    const int N = 1024, D = 64, K = 32;
    FloatRowMat X = make_random(N, D, /*seed=*/0);

    QuantizeConfig cfg;
    cfg.avg_bits = 4.0f;

    saq::IVF ivf(static_cast<size_t>(N), static_cast<size_t>(D),
                 static_cast<size_t>(K), cfg);
    ivf.fit(X, /*apply_pca=*/false, /*K=*/K, /*seed=*/0, /*num_threads=*/1);

    assert(ivf.num_data() == static_cast<size_t>(N));
    assert(ivf.num_dim() == static_cast<size_t>(D));

    std::printf("FitSmoke: OK\n");
}

static void test_decompress_shape_and_finite() {
    const int N = 512, D = 32, K = 16;
    FloatRowMat X = make_random(N, D, /*seed=*/1);

    QuantizeConfig cfg;
    cfg.avg_bits = 4.0f;

    saq::IVF ivf(static_cast<size_t>(N), static_cast<size_t>(D),
                 static_cast<size_t>(K), cfg);
    ivf.fit(X, /*apply_pca=*/false, /*K=*/K, /*seed=*/0, /*num_threads=*/1);

    std::vector<PID> ids = {0, 1, 10, 100, 511};
    FloatRowMat recon = ivf.decompress(ids);

    assert(recon.rows() == static_cast<Eigen::Index>(ids.size()));
    assert(recon.cols() == static_cast<Eigen::Index>(D));
    assert(recon.allFinite());

    std::printf("DecompressShapeAndFinite: OK\n");
}

static float full_reconstruction_mse(bool apply_pca, int seed) {
    const int N = 1024, D = 64, K = 32;
    FloatRowMat X = make_random(N, D, seed);

    QuantizeConfig cfg;
    cfg.avg_bits = 4.0f;

    saq::IVF ivf(static_cast<size_t>(N), static_cast<size_t>(D),
                 static_cast<size_t>(K), cfg);
    ivf.fit(X, apply_pca, /*K=*/K, /*seed=*/0, /*num_threads=*/1);

    std::vector<PID> ids(N);
    std::iota(ids.begin(), ids.end(), 0u);
    FloatRowMat recon = ivf.decompress(ids);

    assert(recon.rows() == static_cast<Eigen::Index>(N));
    assert(recon.cols() == static_cast<Eigen::Index>(D));
    assert(recon.allFinite());

    // Per-element MSE in original (pre-PCA) space.
    return (X - recon).array().square().mean();
}

static void test_decompress_mse_no_pca() {
    float mse = full_reconstruction_mse(/*apply_pca=*/false, /*seed=*/2);
    std::printf("DecompressMSEReasonable_NoPCA: mse=%.6f\n", mse);
    assert(mse < 0.2f);
    std::printf("DecompressMSEReasonable_NoPCA: OK\n");
}

static void test_decompress_mse_with_pca() {
    float mse = full_reconstruction_mse(/*apply_pca=*/true, /*seed=*/3);
    std::printf("DecompressMSEReasonable_WithPCA: mse=%.6f\n", mse);
    assert(mse < 0.2f);
    std::printf("DecompressMSEReasonable_WithPCA: OK\n");
}

int main() {
    test_fit_smoke();
    test_decompress_shape_and_finite();
    test_decompress_mse_no_pca();
    test_decompress_mse_with_pca();
    std::printf("\nAll fit/decompress tests passed!\n");
    return 0;
}
