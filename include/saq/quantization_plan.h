#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

enum class SerializationFormat : uint8_t {
  kBinary = 0,
  kJson = 1
};

struct PCAParams {
  uint32_t input_dim = 0;
  uint32_t output_dim = 0;
  bool enabled = false;
  std::vector<float> mean;      // size: input_dim
  std::vector<float> components; // size: output_dim * input_dim (row-major)
};

struct Segment {
  uint32_t id = 0;
  uint32_t start_dim = 0; // inclusive
  uint32_t dim_count = 0;
  uint32_t bits = 0;      // bits allocated to this segment
};

struct Codebook {
  uint32_t segment_id = 0;
  uint32_t bits = 0;
  uint32_t centroids = 0;  // typically 2^bits
  uint32_t dim_count = 0;
  std::vector<float> data; // size: centroids * dim_count
};

struct QuantizationPlan {
  uint32_t version = 1;
  uint32_t dimension = 0;          // input dimension
  uint32_t total_bits = 0;         // total code length per vector
  uint32_t segment_count = 0;
  uint32_t codebook_count = 0;
  uint32_t seed = 0;               // optional RNG seed for reproducibility
  bool use_pca = false;

  PCAParams pca;
  std::vector<Segment> segments;
  std::vector<Codebook> codebooks;

  bool Validate(std::string* error) const;

  std::vector<uint8_t> SerializeBinary() const;
  bool DeserializeBinary(const std::vector<uint8_t>& data, std::string* error);

  std::string SerializeJson(bool pretty = false) const;
  bool DeserializeJson(const std::string& json, std::string* error);
};

}

