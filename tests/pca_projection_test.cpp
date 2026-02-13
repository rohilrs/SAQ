#include "saq/pca_projection.h"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-4f;

bool ApproxEqual(float a, float b, float eps = kEpsilon) {
  return std::abs(a - b) < eps;
}

float Dot(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float Norm(const float* v, uint32_t dim) {
  return std::sqrt(Dot(v, v, dim));
}

// Test 1: Basic 2D data with clear principal direction
bool TestSimple2D() {
  std::cout << "TestSimple2D: ";

  // Data along y = 2x line with some noise
  // Principal component should be approximately (1/sqrt(5), 2/sqrt(5))
  const uint32_t n = 100;
  const uint32_t dim = 2;
  std::vector<float> data(n * dim);

  std::mt19937 rng(123);
  std::normal_distribution<float> noise(0.0f, 0.1f);

  for (uint32_t i = 0; i < n; ++i) {
    float t = static_cast<float>(i) / n * 10.0f - 5.0f;
    data[i * dim + 0] = t + noise(rng);
    data[i * dim + 1] = 2.0f * t + noise(rng);
  }

  saq::PCAProjection pca;
  if (!pca.Train(data.data(), n, dim, 1)) {
    std::cout << "FAILED - Train failed: " << pca.Error() << "\n";
    return false;
  }

  // First principal component should be close to (1, 2) normalized
  const float* comp = pca.Components();
  float expected_x = 1.0f / std::sqrt(5.0f);
  float expected_y = 2.0f / std::sqrt(5.0f);

  // Component could be flipped (negative), so check absolute values
  if (!ApproxEqual(std::abs(comp[0]), std::abs(expected_x), 0.1f) ||
      !ApproxEqual(std::abs(comp[1]), std::abs(expected_y), 0.1f)) {
    std::cout << "FAILED - Component mismatch: got (" << comp[0] << ", "
              << comp[1] << "), expected ~(+/-" << expected_x << ", +/-"
              << expected_y << ")\n";
    return false;
  }

  std::cout << "OK\n";
  return true;
}

// Test 2: Orthogonality of components
bool TestOrthogonality() {
  std::cout << "TestOrthogonality: ";

  const uint32_t n = 200;
  const uint32_t dim = 5;
  const uint32_t k = 3;
  std::vector<float> data(n * dim);

  std::mt19937 rng(456);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (uint32_t i = 0; i < n * dim; ++i) {
    data[i] = dist(rng);
  }

  saq::PCAProjection pca;
  if (!pca.Train(data.data(), n, dim, k)) {
    std::cout << "FAILED - Train failed: " << pca.Error() << "\n";
    return false;
  }

  const float* comp = pca.Components();

  // Check each component is unit length
  for (uint32_t i = 0; i < k; ++i) {
    float norm = Norm(comp + i * dim, dim);
    if (!ApproxEqual(norm, 1.0f, 0.01f)) {
      std::cout << "FAILED - Component " << i << " not unit length: " << norm
                << "\n";
      return false;
    }
  }

  // Check orthogonality between components
  for (uint32_t i = 0; i < k; ++i) {
    for (uint32_t j = i + 1; j < k; ++j) {
      float dot = Dot(comp + i * dim, comp + j * dim, dim);
      if (!ApproxEqual(dot, 0.0f, 0.05f)) {
        std::cout << "FAILED - Components " << i << " and " << j
                  << " not orthogonal: dot = " << dot << "\n";
        return false;
      }
    }
  }

  std::cout << "OK\n";
  return true;
}

// Test 3: Project and inverse project roundtrip
bool TestRoundtrip() {
  std::cout << "TestRoundtrip: ";

  const uint32_t n = 50;
  const uint32_t dim = 8;
  const uint32_t k = 8;  // Full rank - should be lossless
  std::vector<float> data(n * dim);

  std::mt19937 rng(789);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (uint32_t i = 0; i < n * dim; ++i) {
    data[i] = dist(rng);
  }

  saq::PCAProjection pca;
  if (!pca.Train(data.data(), n, dim, k)) {
    std::cout << "FAILED - Train failed: " << pca.Error() << "\n";
    return false;
  }

  // Take one vector, project and inverse project
  std::vector<float> original(dim);
  std::vector<float> projected(k);
  std::vector<float> reconstructed(dim);

  for (uint32_t i = 0; i < dim; ++i) {
    original[i] = data[i];
  }

  pca.Project(original.data(), projected.data());
  pca.InverseProject(projected.data(), reconstructed.data());

  // With full rank, reconstruction should be nearly perfect
  float max_error = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    float err = std::abs(original[i] - reconstructed[i]);
    max_error = std::max(max_error, err);
  }

  if (max_error > 0.01f) {
    std::cout << "FAILED - Reconstruction error too high: " << max_error
              << "\n";
    return false;
  }

  std::cout << "OK\n";
  return true;
}

// Test 4: Export and import params
bool TestExportImport() {
  std::cout << "TestExportImport: ";

  const uint32_t n = 50;
  const uint32_t dim = 4;
  const uint32_t k = 2;
  std::vector<float> data(n * dim);

  std::mt19937 rng(101112);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (uint32_t i = 0; i < n * dim; ++i) {
    data[i] = dist(rng);
  }

  saq::PCAProjection pca1;
  if (!pca1.Train(data.data(), n, dim, k)) {
    std::cout << "FAILED - Train failed: " << pca1.Error() << "\n";
    return false;
  }

  // Export
  saq::PCAParams params;
  pca1.ExportParams(&params);

  // Import into new PCA
  saq::PCAProjection pca2;
  if (!pca2.ImportParams(params)) {
    std::cout << "FAILED - Import failed: " << pca2.Error() << "\n";
    return false;
  }

  // Both should produce same projections
  std::vector<float> test_vec(dim);
  for (uint32_t i = 0; i < dim; ++i) {
    test_vec[i] = dist(rng);
  }

  std::vector<float> proj1(k), proj2(k);
  pca1.Project(test_vec.data(), proj1.data());
  pca2.Project(test_vec.data(), proj2.data());

  for (uint32_t i = 0; i < k; ++i) {
    if (!ApproxEqual(proj1[i], proj2[i])) {
      std::cout << "FAILED - Projections differ at " << i << ": " << proj1[i]
                << " vs " << proj2[i] << "\n";
      return false;
    }
  }

  std::cout << "OK\n";
  return true;
}

// Test 5: Eigenvalues are in descending order
bool TestEigenvalueOrder() {
  std::cout << "TestEigenvalueOrder: ";

  const uint32_t n = 100;
  const uint32_t dim = 6;
  const uint32_t k = 4;
  std::vector<float> data(n * dim);

  std::mt19937 rng(131415);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (uint32_t i = 0; i < n * dim; ++i) {
    data[i] = dist(rng);
  }

  saq::PCAProjection pca;
  if (!pca.Train(data.data(), n, dim, k)) {
    std::cout << "FAILED - Train failed: " << pca.Error() << "\n";
    return false;
  }

  const auto& eigenvalues = pca.Eigenvalues();
  for (uint32_t i = 1; i < k; ++i) {
    if (eigenvalues[i] > eigenvalues[i - 1] + kEpsilon) {
      std::cout << "FAILED - Eigenvalues not descending: [" << i - 1
                << "]=" << eigenvalues[i - 1] << " < [" << i
                << "]=" << eigenvalues[i] << "\n";
      return false;
    }
  }

  std::cout << "OK\n";
  return true;
}

// Test 6: Large-dimension PCA precision (D=256, full rotation)
bool TestLargeDimPrecision() {
  std::cout << "TestLargeDimPrecision: ";

  const uint32_t n = 500;
  const uint32_t dim = 256;
  const uint32_t k = 256;  // Full rotation
  std::vector<float> data(n * dim);

  std::mt19937 rng(42);

  // Generate data where dim d has variance proportional to 10/(1+d)
  for (uint32_t d = 0; d < dim; ++d) {
    float stddev = std::sqrt(10.0f / (1.0f + d));
    std::normal_distribution<float> dist(0.0f, stddev);
    for (uint32_t i = 0; i < n; ++i) {
      data[i * dim + d] = dist(rng);
    }
  }

  saq::PCAProjection pca;
  if (!pca.Train(data.data(), n, dim, k)) {
    std::cout << "FAILED - Train failed: " << pca.Error() << "\n";
    return false;
  }

  // Check 1: Eigenvalues in strict descending order
  const auto& eigenvalues = pca.Eigenvalues();
  for (uint32_t i = 1; i < k; ++i) {
    if (eigenvalues[i] >= eigenvalues[i - 1]) {
      std::cout << "FAILED - Eigenvalues not strictly descending: [" << i - 1
                << "]=" << eigenvalues[i - 1] << " <= [" << i
                << "]=" << eigenvalues[i] << "\n";
      return false;
    }
  }

  // Check 2: No negative eigenvalues
  for (uint32_t i = 0; i < k; ++i) {
    if (eigenvalues[i] < 0.0f) {
      std::cout << "FAILED - Negative eigenvalue at [" << i
                << "]=" << eigenvalues[i] << "\n";
      return false;
    }
  }

  // Check 3: Orthonormality of component rows (sample first, middle, last)
  const float* comp = pca.Components();
  uint32_t sample_rows[] = {0, k / 2, k - 1};
  for (uint32_t ri = 0; ri < 3; ++ri) {
    uint32_t row = sample_rows[ri];
    // Unit length
    float norm = Norm(comp + row * dim, dim);
    if (!ApproxEqual(norm, 1.0f, 0.01f)) {
      std::cout << "FAILED - Component " << row
                << " not unit length: " << norm << "\n";
      return false;
    }
    // Orthogonal to other sampled rows
    for (uint32_t rj = ri + 1; rj < 3; ++rj) {
      uint32_t other = sample_rows[rj];
      float dot = Dot(comp + row * dim, comp + other * dim, dim);
      if (!ApproxEqual(dot, 0.0f, 0.05f)) {
        std::cout << "FAILED - Components " << row << " and " << other
                  << " not orthogonal: dot = " << dot << "\n";
        return false;
      }
    }
  }

  // Check 4: Full-rotation distance preservation (max relative error < 1%)
  std::vector<float> proj_a(k), proj_b(k);
  float max_rel_error = 0.0f;
  const uint32_t n_pairs = 50;
  std::uniform_int_distribution<uint32_t> idx_dist(0, n - 1);
  for (uint32_t p = 0; p < n_pairs; ++p) {
    uint32_t ia = idx_dist(rng);
    uint32_t ib = idx_dist(rng);
    if (ia == ib) continue;

    const float* va = data.data() + ia * dim;
    const float* vb = data.data() + ib * dim;

    // Original distance
    float dist_orig = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
      float diff = va[d] - vb[d];
      dist_orig += diff * diff;
    }
    dist_orig = std::sqrt(dist_orig);

    // Projected distance
    pca.Project(va, proj_a.data());
    pca.Project(vb, proj_b.data());
    float dist_proj = 0.0f;
    for (uint32_t d = 0; d < k; ++d) {
      float diff = proj_a[d] - proj_b[d];
      dist_proj += diff * diff;
    }
    dist_proj = std::sqrt(dist_proj);

    if (dist_orig > 1e-6f) {
      float rel_error = std::abs(dist_orig - dist_proj) / dist_orig;
      max_rel_error = std::max(max_rel_error, rel_error);
    }
  }

  if (max_rel_error > 0.01f) {
    std::cout << "FAILED - Distance preservation error too high: "
              << max_rel_error * 100.0f << "%\n";
    return false;
  }

  std::cout << "OK\n";
  return true;
}

}  // namespace

int main() {
  int failed = 0;

  if (!TestSimple2D()) ++failed;
  if (!TestOrthogonality()) ++failed;
  if (!TestRoundtrip()) ++failed;
  if (!TestExportImport()) ++failed;
  if (!TestEigenvalueOrder()) ++failed;
  if (!TestLargeDimPrecision()) ++failed;

  if (failed == 0) {
    std::cout << "\nAll PCA tests passed!\n";
    return 0;
  } else {
    std::cout << "\n" << failed << " test(s) failed.\n";
    return 1;
  }
}
