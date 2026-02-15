#include "saq/cluster_data.h"

namespace saq {

void SaqCluData::load(std::ifstream &input) {
    input.read(reinterpret_cast<char *>(short_factors_), shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
    input.read(reinterpret_cast<char *>(short_code_), shortb_code_bytes_ * num_blocks_);
    input.read(reinterpret_cast<char *>(long_code_), longb_code_bytes_ * num_vec_);
    input.read(reinterpret_cast<char *>(long_factors_), num_vec_ * num_segments_ * sizeof(ExFactor));
    input.read(reinterpret_cast<char *>(ids_.data()), ids_.size() * sizeof(PID));
    for (auto &clu : segments_) {
        input.read(reinterpret_cast<char *>(clu.centroid_.data()), clu.centroid_.cols() * sizeof(float));
    }
}

void SaqCluData::save(std::ofstream &output) const {
    output.write(reinterpret_cast<const char *>(short_factors_), shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
    output.write(reinterpret_cast<const char *>(short_code_), shortb_code_bytes_ * num_blocks_);
    output.write(reinterpret_cast<const char *>(long_code_), longb_code_bytes_ * num_vec_);
    output.write(reinterpret_cast<const char *>(long_factors_), num_vec_ * num_segments_ * sizeof(ExFactor));
    output.write(reinterpret_cast<const char *>(ids_.data()), ids_.size() * sizeof(PID));
    for (const auto &clu : segments_) {
        output.write(reinterpret_cast<const char *>(clu.centroid_.data()), clu.centroid_.cols() * sizeof(float));
    }
}

} // namespace saq
