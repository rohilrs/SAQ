#pragma once

/// @file caq_code_adjustment.h
/// @brief Context-Aware Quantization (CAQ) code adjustment for SAQ.
///
/// Implements the iterative code refinement procedure from the SAQ paper.
/// After initial quantization, CAQ adjusts codes by considering the
/// reconstruction error across all segments, improving overall accuracy.

#include "saq/quantization_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace saq {

/// @brief Configuration for CAQ code adjustment.
struct CAQConfig {
  /// Maximum iterations for code refinement.
  uint32_t max_iterations = 10;

  /// Convergence threshold for early stopping.
  /// Stop when relative distortion improvement is below this.
  float convergence_threshold = 1e-4f;

  /// Whether to use beam search for code selection.
  bool use_beam_search = false;

  /// Beam width for beam search (if enabled).
  uint32_t beam_width = 4;

  /// Number of parallel threads (0 = auto).
  uint32_t num_threads = 0;
};

/// @brief Statistics from CAQ adjustment.
struct CAQStats {
  /// Number of iterations performed.
  uint32_t iterations = 0;

  /// Initial total distortion before adjustment.
  float initial_distortion = 0.0f;

  /// Final total distortion after adjustment.
  float final_distortion = 0.0f;

  /// Number of codes that were changed.
  uint64_t codes_changed = 0;

  /// Whether convergence was reached.
  bool converged = false;
};

/// @brief Result of encoding a single vector.
struct EncodedVector {
  /// Quantization codes, one per segment.
  std::vector<uint32_t> codes;

  /// Reconstruction error (squared L2 distance).
  float distortion = 0.0f;
};

/// @brief Context-Aware Quantization code adjuster.
///
/// Given initial quantization codes, refines them by iteratively
/// considering the reconstruction error. For each segment, finds
/// the code that minimizes the total error given the other segments'
/// current codes.
class CAQAdjuster {
 public:
  CAQAdjuster() = default;
  ~CAQAdjuster() = default;

  // Non-copyable, movable
  CAQAdjuster(const CAQAdjuster&) = delete;
  CAQAdjuster& operator=(const CAQAdjuster&) = delete;
  CAQAdjuster(CAQAdjuster&&) = default;
  CAQAdjuster& operator=(CAQAdjuster&&) = default;

  /// @brief Initialize with codebooks from a quantization plan.
  /// @param codebooks Codebooks for each segment.
  /// @param segments Segment definitions.
  /// @return True on success.
  bool Initialize(const std::vector<Codebook>& codebooks,
                  const std::vector<Segment>& segments);

  /// @brief Encode a single vector with initial (greedy) quantization.
  /// @param vector Input vector.
  /// @param dim Vector dimensionality.
  /// @return Encoded vector with codes and distortion.
  EncodedVector EncodeGreedy(const float* vector, uint32_t dim) const;

  /// @brief Refine codes using CAQ adjustment.
  /// @param vector Original input vector.
  /// @param dim Vector dimensionality.
  /// @param initial Initial codes from greedy encoding.
  /// @param config CAQ configuration.
  /// @return Refined encoded vector.
  EncodedVector RefineCAQ(const float* vector, uint32_t dim,
                          const EncodedVector& initial,
                          const CAQConfig& config) const;

  /// @brief Encode and refine a batch of vectors.
  /// @param vectors Row-major matrix of shape (n_vectors, dim).
  /// @param n_vectors Number of vectors.
  /// @param dim Vector dimensionality.
  /// @param config CAQ configuration.
  /// @param stats Optional output statistics.
  /// @return Vector of encoded results.
  std::vector<EncodedVector> EncodeBatch(const float* vectors,
                                          uint32_t n_vectors, uint32_t dim,
                                          const CAQConfig& config,
                                          CAQStats* stats = nullptr) const;

  /// @brief Reconstruct a vector from codes.
  /// @param codes Quantization codes, one per segment.
  /// @param output Output buffer (must have capacity >= total dimensions).
  /// @return True on success.
  bool Reconstruct(const std::vector<uint32_t>& codes, float* output) const;

  /// @brief Compute reconstruction distortion for given codes.
  /// @param vector Original vector.
  /// @param dim Vector dimensionality.
  /// @param codes Quantization codes.
  /// @return Squared L2 distortion.
  float ComputeDistortion(const float* vector, uint32_t dim,
                          const std::vector<uint32_t>& codes) const;

  /// @brief Get total output dimensionality.
  uint32_t TotalDim() const { return total_dim_; }

  /// @brief Get number of segments.
  uint32_t NumSegments() const { return static_cast<uint32_t>(segments_.size()); }

  /// @brief Check if initialized.
  bool IsInitialized() const { return !codebooks_.empty(); }

 private:
  /// @brief Find best code for a segment given fixed codes for other segments.
  /// @param vector Original vector.
  /// @param segment_idx Segment to optimize.
  /// @param current_codes Current codes (will be modified).
  /// @param residual Current residual vector.
  /// @return Best code for this segment.
  uint32_t FindBestCode(const float* vector, uint32_t segment_idx,
                        std::vector<uint32_t>& current_codes,
                        std::vector<float>& residual) const;

  std::vector<Codebook> codebooks_;
  std::vector<Segment> segments_;
  uint32_t total_dim_ = 0;
};

}  // namespace saq
