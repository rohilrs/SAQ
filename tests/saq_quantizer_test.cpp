/// @file saq_quantizer_test.cpp
/// @brief Tests for the main SAQ quantizer (scalar quantization).

#include "saq/saq_quantizer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-4f;

/// @brief Generate random test data.
std::vector<float> GenerateTestData(uint32_t n_vectors, uint32_t dim, uint32_t seed) {
  std::vector<float> data(static_cast<size_t>(n_vectors) * dim);
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (float& v : data) {
    v = dist(rng);
  }

  return data;
}

void TestBasicTraining() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(1000, 32, 42);

  saq::SAQTrainConfig config;
  config.total_bits = 32;

  std::string error = quantizer.Train(data.data(), 1000, 32, config);
  assert(error.empty());
  assert(quantizer.IsTrained());
  assert(quantizer.Dim() == 32);
  assert(quantizer.TotalBits() == 32);

  std::printf("TestBasicTraining: OK\n");
}

void TestEncodeAndDecode() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(500, 64, 123);

  saq::SAQTrainConfig config;
  config.total_bits = 64;

  std::string error = quantizer.Train(data.data(), 500, 64, config);
  if (!error.empty()) {
    std::printf("TestEncodeAndDecode: Training error: %s\n", error.c_str());
  }
  assert(error.empty());

  // Encode a vector
  saq::ScalarEncodedVector encoded;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;

  bool ok = quantizer.Encode(data.data(), encoded, encode_config);
  assert(ok);
  assert(!encoded.codes.empty());
  assert(encoded.v_max > 0.0f);

  // Decode and check reconstruction error
  std::vector<float> reconstructed(64);
  ok = quantizer.Decode(encoded, reconstructed.data());
  assert(ok);

  // Compute reconstruction error
  float error_sq = 0.0f;
  for (uint32_t d = 0; d < 64; ++d) {
    float diff = data[d] - reconstructed[d];
    error_sq += diff * diff;
  }

  // Error should be reasonable (not zero, but not huge)
  assert(error_sq < 64.0f * 2.0f);  // Less than 2.0 average per dimension

  std::printf("TestEncodeAndDecode: OK (reconstruction error: %.4f)\n", error_sq);
}

void TestBatchEncode() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(200, 32, 456);

  saq::SAQTrainConfig config;
  config.total_bits = 32;

  quantizer.Train(data.data(), 200, 32, config);

  // Batch encode
  std::vector<saq::ScalarEncodedVector> encoded_batch;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;

  bool ok = quantizer.EncodeBatch(data.data(), 200, encoded_batch, encode_config);
  assert(ok);
  assert(encoded_batch.size() == 200);

  // Verify by individual encoding
  for (uint32_t v = 0; v < 10; ++v) {
    saq::ScalarEncodedVector single;
    quantizer.Encode(data.data() + v * 32, single, encode_config);

    assert(single.codes.size() == encoded_batch[v].codes.size());
    for (size_t d = 0; d < single.codes.size(); ++d) {
      assert(single.codes[d] == encoded_batch[v].codes[d]);
    }
    assert(std::abs(single.v_max - encoded_batch[v].v_max) < kEpsilon);
  }

  std::printf("TestBatchEncode: OK\n");
}

void TestSearch() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(1000, 32, 789);

  saq::SAQTrainConfig config;
  config.total_bits = 32;

  quantizer.Train(data.data(), 1000, 32, config);

  // Encode all vectors
  std::vector<saq::ScalarEncodedVector> encoded_db;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;
  quantizer.EncodeBatch(data.data(), 1000, encoded_db, encode_config);

  // Search with first vector as query
  const float* query = data.data();
  std::vector<saq::SearchResult> results;
  quantizer.Search(query, encoded_db, 10, results);

  assert(results.size() == 10);

  // Results should be sorted by distance
  for (size_t i = 1; i < results.size(); ++i) {
    assert(results[i].distance >= results[i-1].distance - kEpsilon);
  }

  std::printf("TestSearch: OK (top-1 index: %u, distance: %.4f)\n",
              results[0].index, results[0].distance);
}

void TestSearchAccuracy() {
  saq::SAQQuantizer quantizer;

  // Create clustered data for easier search
  std::mt19937 rng(999);
  std::normal_distribution<float> dist(0.0f, 0.1f);

  const uint32_t n_clusters = 10;
  const uint32_t per_cluster = 100;
  const uint32_t dim = 32;
  const uint32_t n_vectors = n_clusters * per_cluster;

  std::vector<float> data(n_vectors * dim);
  std::vector<float> cluster_centers(n_clusters * dim);

  // Generate cluster centers
  std::uniform_real_distribution<float> center_dist(-5.0f, 5.0f);
  for (float& v : cluster_centers) {
    v = center_dist(rng);
  }

  // Generate points around centers
  for (uint32_t c = 0; c < n_clusters; ++c) {
    for (uint32_t p = 0; p < per_cluster; ++p) {
      uint32_t idx = c * per_cluster + p;
      for (uint32_t d = 0; d < dim; ++d) {
        data[idx * dim + d] = cluster_centers[c * dim + d] + dist(rng);
      }
    }
  }

  // 6 bpd (192 bits / 32 dims): quantization error² ~ 0.09 per vector,
  // well below intra-cluster L2 distance ~ 0.64 (noise stddev 0.1).
  saq::SAQTrainConfig config;
  config.total_bits = 192;

  quantizer.Train(data.data(), n_vectors, dim, config);

  std::vector<saq::ScalarEncodedVector> encoded_db;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;
  quantizer.EncodeBatch(data.data(), n_vectors, encoded_db, encode_config);

  // Test recall: query each vector, check if correct result is in top-k
  int recall_at_1 = 0;
  int recall_at_10 = 0;

  for (uint32_t q = 0; q < 100; ++q) {
    const float* query = data.data() + q * dim;
    std::vector<saq::SearchResult> results;
    quantizer.Search(query, encoded_db, 10, results);

    if (!results.empty() && results[0].index == q) {
      recall_at_1++;
    }

    for (const auto& r : results) {
      if (r.index == q) {
        recall_at_10++;
        break;
      }
    }
  }

  std::printf("TestSearchAccuracy: OK (R@1: %d%%, R@10: %d%%)\n",
              recall_at_1, recall_at_10);

  // Expect high recall for this easy case (6 bpd on well-separated clusters)
  assert(recall_at_10 >= 80);
}

void TestWithPCA() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(500, 128, 111);

  saq::SAQTrainConfig config;
  config.total_bits = 64;
  config.use_pca = true;
  config.pca_dim = 64;

  std::string error = quantizer.Train(data.data(), 500, 128, config);
  assert(error.empty());
  assert(quantizer.Dim() == 128);
  assert(quantizer.WorkingDim() == 64);

  // Encode and decode
  saq::ScalarEncodedVector encoded;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;

  bool ok = quantizer.Encode(data.data(), encoded, encode_config);
  assert(ok);

  std::vector<float> reconstructed(128);
  ok = quantizer.Decode(encoded, reconstructed.data());
  assert(ok);

  std::printf("TestWithPCA: OK\n");
}

void TestPlanSerialization() {
  saq::SAQQuantizer quantizer1;

  auto data = GenerateTestData(300, 32, 222);

  saq::SAQTrainConfig config;
  config.total_bits = 32;

  quantizer1.Train(data.data(), 300, 32, config);

  // Encode a test vector
  saq::ScalarEncodedVector encoded1;
  saq::SAQEncodeConfig encode_config;
  encode_config.use_caq = false;
  quantizer1.Encode(data.data(), encoded1, encode_config);

  // Save and load plan
  auto binary = quantizer1.Plan().SerializeBinary();
  assert(!binary.empty());

  saq::QuantizationPlan loaded_plan;
  std::string error;
  bool ok = loaded_plan.DeserializeBinary(binary, &error);
  assert(ok);

  saq::SAQQuantizer quantizer2;
  std::string load_error = quantizer2.LoadPlan(loaded_plan);
  assert(load_error.empty());
  assert(quantizer2.IsTrained());

  // Encode same vector with loaded quantizer
  saq::ScalarEncodedVector encoded2;
  quantizer2.Encode(data.data(), encoded2, encode_config);

  // Codes should match
  assert(encoded1.codes.size() == encoded2.codes.size());
  for (size_t d = 0; d < encoded1.codes.size(); ++d) {
    assert(encoded1.codes[d] == encoded2.codes[d]);
  }
  assert(std::abs(encoded1.v_max - encoded2.v_max) < kEpsilon);

  std::printf("TestPlanSerialization: OK\n");
}

void TestCAQRefinement() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(500, 64, 444);

  saq::SAQTrainConfig config;
  config.total_bits = 64;

  quantizer.Train(data.data(), 500, 64, config);

  // Compare with and without CAQ
  saq::ScalarEncodedVector encoded_no_caq;
  saq::ScalarEncodedVector encoded_caq;

  saq::SAQEncodeConfig config_no_caq;
  config_no_caq.use_caq = false;

  saq::SAQEncodeConfig config_caq;
  config_caq.use_caq = true;
  config_caq.caq_config.num_rounds = 10;

  const float* test_vec = data.data();
  quantizer.Encode(test_vec, encoded_no_caq, config_no_caq);
  quantizer.Encode(test_vec, encoded_caq, config_caq);

  // Decode both
  std::vector<float> recon_no_caq(64);
  std::vector<float> recon_caq(64);
  quantizer.Decode(encoded_no_caq, recon_no_caq.data());
  quantizer.Decode(encoded_caq, recon_caq.data());

  // Compute cosine similarity (CAQ optimizes for cosine)
  float dot_no_caq = 0.0f, dot_caq = 0.0f;
  float norm_orig = 0.0f, norm_no_caq = 0.0f, norm_caq = 0.0f;
  for (uint32_t d = 0; d < 64; ++d) {
    dot_no_caq += test_vec[d] * recon_no_caq[d];
    dot_caq += test_vec[d] * recon_caq[d];
    norm_orig += test_vec[d] * test_vec[d];
    norm_no_caq += recon_no_caq[d] * recon_no_caq[d];
    norm_caq += recon_caq[d] * recon_caq[d];
  }

  float cos_no_caq = dot_no_caq / (std::sqrt(norm_orig) * std::sqrt(norm_no_caq) + 1e-10f);
  float cos_caq = dot_caq / (std::sqrt(norm_orig) * std::sqrt(norm_caq) + 1e-10f);

  // CAQ should be at least as good (in cosine similarity)
  assert(cos_caq >= cos_no_caq - kEpsilon);

  std::printf("TestCAQRefinement: OK (cosine without CAQ: %.4f, with CAQ: %.4f)\n",
              cos_no_caq, cos_caq);
}

void TestTransformQuery() {
  saq::SAQQuantizer quantizer;

  auto data = GenerateTestData(500, 64, 555);

  saq::SAQTrainConfig config;
  config.total_bits = 64;
  config.use_segment_rotation = true;

  quantizer.Train(data.data(), 500, 64, config);

  // Transform query should succeed
  std::vector<float> rotated(quantizer.WorkingDim());
  bool ok = quantizer.TransformQuery(data.data(), rotated.data());
  assert(ok);

  // Output should not be all zeros
  float sum = 0.0f;
  for (float v : rotated) {
    sum += std::abs(v);
  }
  assert(sum > 0.0f);

  std::printf("TestTransformQuery: OK\n");
}

}  // namespace

int main() {
  TestBasicTraining();
  TestEncodeAndDecode();
  TestBatchEncode();
  TestSearch();
  TestSearchAccuracy();
  TestWithPCA();
  TestPlanSerialization();
  TestCAQRefinement();
  TestTransformQuery();

  std::printf("\nAll SAQ quantizer tests passed!\n");
  return 0;
}
