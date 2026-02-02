/// @file quantization_plan.cpp
/// @brief Implementation of QuantizationPlan serialization.

#include "saq/quantization_plan.h"

#include <cstring>
#include <tao/json.hpp>

namespace saq {

namespace {

/// Binary format magic number: "PQSA" in little-endian.
constexpr uint32_t kBinaryMagic = 0x53515150;

void WriteU32(std::vector<uint8_t>* out, uint32_t v) {
  out->push_back(static_cast<uint8_t>(v & 0xFFu));
  out->push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
  out->push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
  out->push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

/// Write a single byte.
void WriteU8(std::vector<uint8_t>* out, uint8_t v) {
  out->push_back(v);
}

/// Read a 32-bit unsigned integer in little-endian format.
bool ReadU32(const std::vector<uint8_t>& in, size_t* offset, uint32_t* v) {
  if (*offset + 4 > in.size()) { return false; }
  *v = static_cast<uint32_t>(in[*offset]) |
       (static_cast<uint32_t>(in[*offset + 1]) << 8) |
       (static_cast<uint32_t>(in[*offset + 2]) << 16) |
       (static_cast<uint32_t>(in[*offset + 3]) << 24);
  *offset += 4;
  return true;
}

/// Read a single byte.
bool ReadU8(const std::vector<uint8_t>& in, size_t* offset, uint8_t* v) {
  if (*offset + 1 > in.size()) { return false; }
  *v = in[*offset];
  *offset += 1;
  return true;
}

/// Write a 32-bit float as a little-endian uint32.
void WriteF32(std::vector<uint8_t>* out, float v) {
  static_assert(sizeof(float) == 4, "float must be 4 bytes");
  uint32_t u;
  std::memcpy(&u, &v, sizeof(float));
  WriteU32(out, u);
}

/// Read a 32-bit float from little-endian uint32.
bool ReadF32(const std::vector<uint8_t>& in, size_t* offset, float* v) {
  uint32_t u = 0;
  if (!ReadU32(in, offset, &u)) { return false; }
  std::memcpy(v, &u, sizeof(float));
  return true;
}

/// Write a length-prefixed array of floats.
void WriteF32Array(std::vector<uint8_t>* out, const std::vector<float>& data) {
  WriteU32(out, static_cast<uint32_t>(data.size()));
  for (float f : data) {
    WriteF32(out, f);
  }
}

/// Read a length-prefixed array of floats.
bool ReadF32Array(const std::vector<uint8_t>& in, size_t* offset,
                  std::vector<float>* data) {
  uint32_t count = 0;
  if (!ReadU32(in, offset, &count)) { return false; }
  data->clear();
  data->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    float f = 0.0f;
    if (!ReadF32(in, offset, &f)) { return false; }
    data->push_back(f);
  }
  return true;
}

// --- JSON serialization helpers ---

/// Convert float vector to JSON array of doubles.
tao::json::value FloatVecToJson(const std::vector<float>& vec) {
  tao::json::value arr = tao::json::empty_array;
  for (float f : vec) {
    arr.emplace_back(static_cast<double>(f));
  }
  return arr;
}

/// Convert JSON array of numbers to float vector.
std::vector<float> JsonToFloatVec(const tao::json::value& v) {
  std::vector<float> result;
  if (v.is_array()) {
    result.reserve(v.get_array().size());
    for (const auto& elem : v.get_array()) {
      result.push_back(static_cast<float>(elem.as<double>()));
    }
  }
  return result;
}

/// Serialize PCAParams to JSON object.
tao::json::value ToJson(const PCAParams& p) {
  tao::json::value v = tao::json::empty_object;
  v["input_dim"] = static_cast<std::uint64_t>(p.input_dim);
  v["output_dim"] = static_cast<std::uint64_t>(p.output_dim);
  v["enabled"] = p.enabled;
  v["mean"] = FloatVecToJson(p.mean);
  v["components"] = FloatVecToJson(p.components);
  return v;
}

/// Serialize Segment to JSON object.
tao::json::value ToJson(const Segment& s) {
  tao::json::value v = tao::json::empty_object;
  v["id"] = static_cast<std::uint64_t>(s.id);
  v["start_dim"] = static_cast<std::uint64_t>(s.start_dim);
  v["dim_count"] = static_cast<std::uint64_t>(s.dim_count);
  v["bits"] = static_cast<std::uint64_t>(s.bits);
  return v;
}

/// Serialize Codebook to JSON object.
tao::json::value ToJson(const Codebook& c) {
  tao::json::value v = tao::json::empty_object;
  v["segment_id"] = static_cast<std::uint64_t>(c.segment_id);
  v["bits"] = static_cast<std::uint64_t>(c.bits);
  v["centroids"] = static_cast<std::uint64_t>(c.centroids);
  v["dim_count"] = static_cast<std::uint64_t>(c.dim_count);
  v["data"] = FloatVecToJson(c.data);
  return v;
}

/// Deserialize PCAParams from JSON object.
bool FromJson(const tao::json::value& v, PCAParams* p) {
  if (!v.is_object()) { return false; }
  p->input_dim = static_cast<uint32_t>(v.at("input_dim").as<std::uint64_t>());
  p->output_dim = static_cast<uint32_t>(v.at("output_dim").as<std::uint64_t>());
  p->enabled = v.at("enabled").as<bool>();
  p->mean = JsonToFloatVec(v.at("mean"));
  p->components = JsonToFloatVec(v.at("components"));
  return true;
}

/// Deserialize Segment from JSON object.
bool FromJson(const tao::json::value& v, Segment* s) {
  if (!v.is_object()) { return false; }
  s->id = static_cast<uint32_t>(v.at("id").as<std::uint64_t>());
  s->start_dim = static_cast<uint32_t>(v.at("start_dim").as<std::uint64_t>());
  s->dim_count = static_cast<uint32_t>(v.at("dim_count").as<std::uint64_t>());
  s->bits = static_cast<uint32_t>(v.at("bits").as<std::uint64_t>());
  return true;
}

/// Deserialize Codebook from JSON object.
bool FromJson(const tao::json::value& v, Codebook* c) {
  if (!v.is_object()) { return false; }
  c->segment_id = static_cast<uint32_t>(v.at("segment_id").as<std::uint64_t>());
  c->bits = static_cast<uint32_t>(v.at("bits").as<std::uint64_t>());
  c->centroids = static_cast<uint32_t>(v.at("centroids").as<std::uint64_t>());
  c->dim_count = static_cast<uint32_t>(v.at("dim_count").as<std::uint64_t>());
  c->data = JsonToFloatVec(v.at("data"));
  return true;
}

}  // anonymous namespace

bool QuantizationPlan::Validate(std::string* error) const {
  if (dimension == 0) {
    if (error) { *error = "dimension must be > 0"; }
    return false;
  }
  if (segment_count != segments.size()) {
    if (error) { *error = "segment_count does not match segments size"; }
    return false;
  }
  if (codebook_count != codebooks.size()) {
    if (error) { *error = "codebook_count does not match codebooks size"; }
    return false;
  }
  return true;
}

std::vector<uint8_t> QuantizationPlan::SerializeBinary() const {
  std::vector<uint8_t> out;
  out.reserve(256);

  WriteU32(&out, kBinaryMagic);
  WriteU32(&out, version);
  WriteU32(&out, dimension);
  WriteU32(&out, total_bits);
  WriteU32(&out, segment_count);
  WriteU32(&out, codebook_count);
  WriteU32(&out, seed);
  WriteU8(&out, use_pca ? 1 : 0);

  WriteU32(&out, pca.input_dim);
  WriteU32(&out, pca.output_dim);
  WriteU8(&out, pca.enabled ? 1 : 0);
  WriteF32Array(&out, pca.mean);
  WriteF32Array(&out, pca.components);

  WriteU32(&out, static_cast<uint32_t>(segments.size()));
  for (const auto& s : segments) {
    WriteU32(&out, s.id);
    WriteU32(&out, s.start_dim);
    WriteU32(&out, s.dim_count);
    WriteU32(&out, s.bits);
  }

  WriteU32(&out, static_cast<uint32_t>(codebooks.size()));
  for (const auto& c : codebooks) {
    WriteU32(&out, c.segment_id);
    WriteU32(&out, c.bits);
    WriteU32(&out, c.centroids);
    WriteU32(&out, c.dim_count);
    WriteF32Array(&out, c.data);
  }

  return out;
}

bool QuantizationPlan::DeserializeBinary(const std::vector<uint8_t>& data, std::string* error) {
  size_t offset = 0;
  uint32_t magic = 0;
  if (!ReadU32(data, &offset, &magic) || magic != kBinaryMagic) {
    if (error) { *error = "invalid binary magic"; }
    return false;
  }

  if (!ReadU32(data, &offset, &version) ||
      !ReadU32(data, &offset, &dimension) ||
      !ReadU32(data, &offset, &total_bits) ||
      !ReadU32(data, &offset, &segment_count) ||
      !ReadU32(data, &offset, &codebook_count) ||
      !ReadU32(data, &offset, &seed)) {
    if (error) { *error = "truncated header"; }
    return false;
  }

  uint8_t use_pca_u8 = 0;
  if (!ReadU8(data, &offset, &use_pca_u8)) {
    if (error) { *error = "truncated use_pca"; }
    return false;
  }
  use_pca = (use_pca_u8 != 0);

  uint8_t pca_enabled_u8 = 0;
  if (!ReadU32(data, &offset, &pca.input_dim) ||
      !ReadU32(data, &offset, &pca.output_dim) ||
      !ReadU8(data, &offset, &pca_enabled_u8)) {
    if (error) { *error = "truncated pca header"; }
    return false;
  }
  pca.enabled = (pca_enabled_u8 != 0);
  if (!ReadF32Array(data, &offset, &pca.mean) ||
      !ReadF32Array(data, &offset, &pca.components)) {
    if (error) { *error = "truncated pca data"; }
    return false;
  }

  uint32_t seg_count = 0;
  if (!ReadU32(data, &offset, &seg_count)) {
    if (error) { *error = "truncated segments count"; }
    return false;
  }
  segments.clear();
  segments.reserve(seg_count);
  for (uint32_t i = 0; i < seg_count; ++i) {
    Segment s;
    if (!ReadU32(data, &offset, &s.id) ||
        !ReadU32(data, &offset, &s.start_dim) ||
        !ReadU32(data, &offset, &s.dim_count) ||
        !ReadU32(data, &offset, &s.bits)) {
      if (error) { *error = "truncated segment entry"; }
      return false;
    }
    segments.push_back(std::move(s));
  }

  uint32_t cb_count = 0;
  if (!ReadU32(data, &offset, &cb_count)) {
    if (error) { *error = "truncated codebooks count"; }
    return false;
  }
  codebooks.clear();
  codebooks.reserve(cb_count);
  for (uint32_t i = 0; i < cb_count; ++i) {
    Codebook c;
    if (!ReadU32(data, &offset, &c.segment_id) ||
        !ReadU32(data, &offset, &c.bits) ||
        !ReadU32(data, &offset, &c.centroids) ||
        !ReadU32(data, &offset, &c.dim_count)) {
      if (error) { *error = "truncated codebook entry header"; }
      return false;
    }
    if (!ReadF32Array(data, &offset, &c.data)) {
      if (error) { *error = "truncated codebook data"; }
      return false;
    }
    codebooks.push_back(std::move(c));
  }

  if (segment_count != segments.size()) {
    segment_count = static_cast<uint32_t>(segments.size());
  }
  if (codebook_count != codebooks.size()) {
    codebook_count = static_cast<uint32_t>(codebooks.size());
  }

  return Validate(error);
}

std::string QuantizationPlan::SerializeJson(bool pretty) const {
  tao::json::value v = tao::json::empty_object;
  v["version"] = static_cast<std::uint64_t>(version);
  v["dimension"] = static_cast<std::uint64_t>(dimension);
  v["total_bits"] = static_cast<std::uint64_t>(total_bits);
  v["segment_count"] = static_cast<std::uint64_t>(segment_count);
  v["codebook_count"] = static_cast<std::uint64_t>(codebook_count);
  v["seed"] = static_cast<std::uint64_t>(seed);
  v["use_pca"] = use_pca;
  v["pca"] = ToJson(pca);

  tao::json::value segs = tao::json::empty_array;
  for (const auto& s : segments) {
    segs.emplace_back(ToJson(s));
  }
  v["segments"] = std::move(segs);

  tao::json::value cbs = tao::json::empty_array;
  for (const auto& c : codebooks) {
    cbs.emplace_back(ToJson(c));
  }
  v["codebooks"] = std::move(cbs);

  return pretty ? tao::json::to_string(v, 2) : tao::json::to_string(v);
}

bool QuantizationPlan::DeserializeJson(const std::string& json, std::string* error) {
  try {
    const auto v = tao::json::from_string(json);
    if (!v.is_object()) {
      if (error) { *error = "root must be a JSON object"; }
      return false;
    }

    version = static_cast<uint32_t>(v.at("version").as<std::uint64_t>());
    dimension = static_cast<uint32_t>(v.at("dimension").as<std::uint64_t>());
    total_bits = static_cast<uint32_t>(v.at("total_bits").as<std::uint64_t>());
    segment_count = static_cast<uint32_t>(v.at("segment_count").as<std::uint64_t>());
    codebook_count = static_cast<uint32_t>(v.at("codebook_count").as<std::uint64_t>());
    seed = static_cast<uint32_t>(v.at("seed").as<std::uint64_t>());
    use_pca = v.at("use_pca").as<bool>();

    if (!FromJson(v.at("pca"), &pca)) {
      if (error) { *error = "invalid pca object"; }
      return false;
    }

    segments.clear();
    for (const auto& s : v.at("segments").get_array()) {
      Segment seg;
      if (!FromJson(s, &seg)) {
        if (error) { *error = "invalid segment entry"; }
        return false;
      }
      segments.push_back(std::move(seg));
    }

    codebooks.clear();
    for (const auto& c : v.at("codebooks").get_array()) {
      Codebook cb;
      if (!FromJson(c, &cb)) {
        if (error) { *error = "invalid codebook entry"; }
        return false;
      }
      codebooks.push_back(std::move(cb));
    }

    if (segment_count != segments.size()) {
      segment_count = static_cast<uint32_t>(segments.size());
    }
    if (codebook_count != codebooks.size()) {
      codebook_count = static_cast<uint32_t>(codebooks.size());
    }

    return Validate(error);
  } catch (const std::exception& e) {
    if (error) { *error = e.what(); }
    return false;
  }
}

}  // namespace saq
