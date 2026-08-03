#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_utils.cuh"

namespace saq::gpu {

// Max dims per lane: ceil(D_seg_max / 32). For D_seg up to 896, this is 28.
constexpr int kMaxDimsPerLane = 32;

// Broadcast a double from a specific warp lane (warp_broadcast_double is lane-0 only).
__device__ __forceinline__ double shfl_double_from(double v, int src) {
    int2 t = *reinterpret_cast<int2*>(&v);
    t.x = __shfl_sync(0xffffffffu, t.x, src);
    t.y = __shfl_sync(0xffffffffu, t.y, src);
    return *reinterpret_cast<double*>(&t);
}

// ============================================================================
// Fused CAQ Encode: subtract rotated centroid + encode + pack short/long codes
// ============================================================================

__global__ void kernel_fused_caq_encode(
    const float* __restrict__ d_vectors_rotated,  // [N x D_seg] GEMM output on raw vectors
    const float* __restrict__ d_rotated_centroids, // [K x D_seg]
    const uint32_t* __restrict__ d_cluster_ids,
    float* __restrict__ d_o_l2norm,
    float* __restrict__ d_fac_rescale,
    float* __restrict__ d_fac_error,
    float* __restrict__ d_ip_cent_oa,
    uint8_t* __restrict__ d_short_raw,   // [N x D_seg/8]
    uint8_t* __restrict__ d_long_raw,    // [N x long_bytes_per_vec]
    size_t D_seg,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB,
    int caq_sequential)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if ((size_t)warp_id >= N) return;

    uint32_t cid = d_cluster_ids[warp_id];

    // Each lane handles a contiguous chunk of dimensions
    size_t chunk = (D_seg + 31) / 32;
    size_t start = lane_id * chunk;
    size_t end = min(start + chunk, D_seg);
    size_t my_dims = (end > start) ? end - start : 0;

    // L1: Compute residual = rotated_vector - rotated_centroid (inline)
    const float* vec_rot = d_vectors_rotated + (size_t)warp_id * D_seg;
    const float* cent_rot = d_rotated_centroids + (size_t)cid * D_seg;

    // Local storage for residuals and codes (in registers)
    float local_vec[kMaxDimsPerLane];
    int local_codes[kMaxDimsPerLane];

    for (size_t i = 0; i < my_dims; ++i) {
        local_vec[i] = vec_rot[start + i] - cent_rot[start + i];
    }

    if (num_bits == 0) {
        // Zero-bit segment: just compute o_l2norm from residual
        double partial_l2 = 0.0;
        for (size_t i = 0; i < my_dims; ++i)
            partial_l2 += (double)local_vec[i] * local_vec[i];
        double total_l2 = warp_reduce_sum_double(partial_l2);
        if (lane_id == 0) {
            d_o_l2norm[warp_id] = sqrtf((float)total_l2);
            d_fac_rescale[warp_id] = 0.0f;
            d_fac_error[warp_id] = 0.0f;
            d_ip_cent_oa[warp_id] = 0.0f;
        }
        return;
    }

    // ---- Step 1: Compute v_max ----
    float local_max = 0.0f;
    for (size_t i = 0; i < my_dims; ++i)
        local_max = fmaxf(local_max, fabsf(local_vec[i]));

    float v_mx = warp_reduce_max(local_max);
    v_mx = warp_broadcast(v_mx);
    float v_mi = -v_mx;
    float delta = (v_mx - v_mi) / (code_max + 1);

    // ---- Step 2: Initial quantization ----
    double partial_ip_o_code = 0.0;
    uint64_t partial_code_l2sqr = 0;
    int partial_code_sum = 0;
    double partial_vec_sum = 0.0;
    double partial_o_l2sqr = 0.0;

    for (size_t i = 0; i < my_dims; ++i) {
        float o = local_vec[i];
        partial_o_l2sqr += (double)o * o;
        partial_vec_sum += o;

        int c;
        if (delta > 0.0f) {
            c = (int)floorf((o - v_mi) / delta);
            c = min(c, (int)code_max);
            c = max(c, 0);
        } else {
            c = 0;
        }
        local_codes[i] = c;
        partial_ip_o_code += (double)c * o;
        partial_code_l2sqr += (uint64_t)c * c;
        partial_code_sum += c;
    }

    double ip_o_code = warp_reduce_sum_double(partial_ip_o_code);
    double vec_sum = warp_reduce_sum_double(partial_vec_sum);
    double o_l2sqr = warp_reduce_sum_double(partial_o_l2sqr);
    double code_l2sqr_d = warp_reduce_sum_double((double)partial_code_l2sqr);
    double code_sum_d = warp_reduce_sum_double((double)partial_code_sum);

    ip_o_code = warp_broadcast_double(ip_o_code);
    vec_sum = warp_broadcast_double(vec_sum);
    o_l2sqr = warp_broadcast_double(o_l2sqr);
    code_l2sqr_d = warp_broadcast_double(code_l2sqr_d);
    code_sum_d = warp_broadcast_double(code_sum_d);

    double ip_o_oa = ip_o_code * delta + (v_mi + 0.5 * delta) * vec_sum;
    double oa_l2sqr = delta * delta * code_l2sqr_d
                    + (delta * delta + 2.0 * delta * v_mi) * code_sum_d
                    + (0.25 * delta * delta + delta * v_mi + (double)v_mi * v_mi) * D_seg;

    // ---- Step 3: Code adjustment ----
    if (caq_adj_rd_lmt && oa_l2sqr > 0.0 && delta > 0.0f) {
        double re_eps = (double)caq_adj_eps * oa_l2sqr;

      if (caq_sequential) {
        // Faithful sequential Gauss-Seidel — bit-identical order to the reference CPU
        // CAQEncoder::code_adjustment. Dims are visited in GLOBAL order; the owning
        // lane adjusts its dim using the up-to-date running ip_o_oa / oa_l2sqr, then
        // broadcasts them to the whole warp so the next dim sees the change. re_eps is
        // held constant across rounds (matching the CPU), and each round ends with an
        // exact correction (kills float drift). Slower than block-Jacobi (31 lanes idle
        // per dim) but reproduces the reference encoder's local optimum.
        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int round_adj = 0;
            for (size_t d = 0; d < D_seg; ++d) {
                int owner = (int)(d / chunk);
                int adj = 0;
                if (lane_id == owner) {
                    size_t i = d - start;
                    float o = local_vec[i];
                    int c = local_codes[i];
                    double oa = (c + 0.5) * delta + v_mi;
                    double oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                    double ip_delta = delta * o;
                    while (c < (int)code_max) {
                        double new_q = oa + delta;
                        double new_length = oa_l2sqr_tmp + new_q * new_q;
                        double new_ip = ip_o_oa + ip_delta;
                        if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr) break;
                        c++; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length; adj++;
                    }
                    while (c > 0) {
                        double new_q = oa - delta;
                        double new_length = oa_l2sqr_tmp + new_q * new_q;
                        double new_ip = ip_o_oa - ip_delta;
                        if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr) break;
                        c--; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length; adj++;
                    }
                    local_codes[i] = c;
                }
                ip_o_oa = shfl_double_from(ip_o_oa, owner);
                oa_l2sqr = shfl_double_from(oa_l2sqr, owner);
                round_adj += __shfl_sync(0xffffffffu, adj, owner);
            }
            if (round_adj == 0) break;
            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                double q = (local_codes[i] + 0.5) * delta + v_mi;
                corr_ip += q * o; corr_oa_l2 += q * q;
            }
            oa_l2sqr = warp_broadcast_double(warp_reduce_sum_double(corr_oa_l2));
            ip_o_oa = warp_broadcast_double(warp_reduce_sum_double(corr_ip));
        }
      } else {
        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int local_adj_cnt = 0;

            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                int c = local_codes[i];
                double oa = (c + 0.5) * delta + v_mi;
                double oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;

                while (c < (int)code_max) {
                    double new_q = oa + delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa + ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                while (c > 0) {
                    double new_q = oa - delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                local_codes[i] = c;
            }

            int total_adj = warp_reduce_sum_int(local_adj_cnt);
            total_adj = warp_broadcast_int(total_adj);
            if (total_adj == 0) break;

            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                double q = (local_codes[i] + 0.5) * delta + v_mi;
                corr_ip += q * o;
                corr_oa_l2 += q * q;
            }
            oa_l2sqr = warp_reduce_sum_double(corr_oa_l2);
            ip_o_oa = warp_reduce_sum_double(corr_ip);
            oa_l2sqr = warp_broadcast_double(oa_l2sqr);
            ip_o_oa = warp_broadcast_double(ip_o_oa);
            re_eps = (double)caq_adj_eps * oa_l2sqr;
        }
      }
    }

    // ---- Step 3b: DownUpSample ----
    if (caq_ori_qB > 0) {
        int sampled_rshift = caq_ori_qB - (int)num_bits;
        delta *= (float)(1 << sampled_rshift);
        double new_ip = 0.0, new_oa_l2 = 0.0;
        for (size_t i = 0; i < my_dims; ++i) {
            local_codes[i] >>= sampled_rshift;
            float o = local_vec[i];
            double q = (local_codes[i] + 0.5) * delta + v_mi;
            new_ip += q * o;
            new_oa_l2 += q * q;
        }
        ip_o_oa = warp_reduce_sum_double(new_ip);
        oa_l2sqr = warp_reduce_sum_double(new_oa_l2);
        ip_o_oa = warp_broadcast_double(ip_o_oa);
        oa_l2sqr = warp_broadcast_double(oa_l2sqr);
    }

    // ---- Step 4: Compute factors ----
    double scale_rate = (v_mx > 0.0f) ? 1.0 / v_mx : 0.0;
    // rescale_vmx_to1: CPU encoder multiplies fac_rescale by v_mx after computing it
    float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa * v_mx) : 0.0f;
    float o_l2norm = sqrtf((float)o_l2sqr);

    constexpr float kConstEpsilon = 1.9f;
    float fac_error = 0.0f;
    if (ip_o_oa > 0.0 && D_seg > 1) {
        fac_error = (float)(o_l2sqr * kConstEpsilon *
            sqrt(((o_l2sqr * oa_l2sqr) / (ip_o_oa * ip_o_oa) - 1.0) / (D_seg - 1)));
    }

    // ---- Step 5: ip_cent_oa ----
    float ip_c_oa = 0.0f;
    if (num_bits > 1) {
        double scaled_delta = delta * scale_rate;
        double scaled_vmi = v_mi * scale_rate;
        double partial_ip = 0.0;
        for (size_t i = 0; i < my_dims; ++i) {
            double oa_d = (local_codes[i] + 0.5) * scaled_delta + scaled_vmi;
            partial_ip += cent_rot[start + i] * oa_d;
        }
        double total_ip = warp_reduce_sum_double(partial_ip);
        ip_c_oa = (float)warp_broadcast_double(total_ip);
    }

    // ---- Write scalar outputs (lane 0 only) ----
    if (lane_id == 0) {
        d_o_l2norm[warp_id] = o_l2norm;
        d_fac_rescale[warp_id] = fac_rescale;
        d_fac_error[warp_id] = fac_error;
        d_ip_cent_oa[warp_id] = ip_c_oa;
    }

    // ---- L2: Pack short codes (descending bit order: dim 0 → bit 7) ----
    if (num_bits > 0 && d_short_raw) {
        size_t short_bytes_per_vec = D_seg / 8;
        // Each lane packs its contiguous chunk of dims into bytes
        // Lane handles dims [start, end). Byte range: [start/8, ceil(end/8))
        for (size_t byte_idx = start / 8; byte_idx < (end + 7) / 8 && byte_idx < short_bytes_per_vec; ++byte_idx) {
            uint8_t byte_val = 0;
            for (int b = 0; b < 8; ++b) {
                size_t d = byte_idx * 8 + b;
                if (d >= start && d < end && d < D_seg) {
                    int c = local_codes[d - start];
                    byte_val |= ((c >> ((int)num_bits - 1)) & 1) << (7 - b);
                }
            }
            // If this byte straddles two lanes' chunks, use atomicOr
            size_t byte_start_dim = byte_idx * 8;
            size_t byte_end_dim = byte_start_dim + 8;
            if (byte_start_dim >= start && byte_end_dim <= end) {
                // Entire byte owned by this lane — direct write
                d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = byte_val;
            } else {
                // Byte straddles boundary — use atomicOr
                // First zero the byte (only the lane that owns the first dim does this)
                if (byte_start_dim == start || start == 0) {
                    d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = 0;
                }
                __syncwarp();
                if (byte_val != 0) {
                    atomicOr((unsigned int*)(d_short_raw + (size_t)warp_id * short_bytes_per_vec + (byte_idx & ~3u)),
                             (unsigned int)byte_val << (8 * (byte_idx & 3u)));
                }
            }
        }
    }

    // ---- L2: Pack long codes ----
    if (num_bits > 1 && d_long_raw) {
        size_t ex_bits = num_bits - 1;
        size_t long_bytes_per_vec = D_seg * ex_bits / 8;
        uint8_t* out = d_long_raw + (size_t)warp_id * long_bytes_per_vec;

        // Each lane writes its dims' lower (num_bits-1) bits
        // Bit position for dim d in the compacted stream: d * ex_bits
        for (size_t i = 0; i < my_dims; ++i) {
            size_t d = start + i;
            int c = local_codes[i] & ((1 << ex_bits) - 1); // lower bits
            size_t bit_offset = d * ex_bits;

            for (size_t b = 0; b < ex_bits; ++b) {
                size_t global_bit = bit_offset + b;
                size_t byte_pos = global_bit / 8;
                size_t bit_pos = global_bit % 8;
                if ((c >> b) & 1) {
                    atomicOr((unsigned int*)(out + (byte_pos & ~3u)),
                             1u << (8 * (byte_pos & 3u) + bit_pos));
                }
            }
        }
    }
}

// No-rotation variant: reads raw vectors + centroids, subtracts inline
__global__ void kernel_fused_caq_encode_no_rotation(
    const float* __restrict__ d_vectors,
    const float* __restrict__ d_centroids,
    const uint32_t* __restrict__ d_cluster_ids,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    float* __restrict__ d_o_l2norm,
    float* __restrict__ d_fac_rescale,
    float* __restrict__ d_fac_error,
    float* __restrict__ d_ip_cent_oa,
    uint8_t* __restrict__ d_short_raw,
    uint8_t* __restrict__ d_long_raw,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB)
{
    // Same as fused kernel but reads from full vectors with offset
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if ((size_t)warp_id >= N) return;

    uint32_t cid = d_cluster_ids[warp_id];
    size_t chunk = (D_seg + 31) / 32;
    size_t start = lane_id * chunk;
    size_t end = min(start + chunk, D_seg);
    size_t my_dims = (end > start) ? end - start : 0;

    const float* vec_base = d_vectors + (size_t)warp_id * D_total + seg_offset;
    const float* cent_base = d_centroids + (size_t)cid * D_total + seg_offset;

    float local_vec[kMaxDimsPerLane];
    for (size_t i = 0; i < my_dims; ++i)
        local_vec[i] = vec_base[start + i] - cent_base[start + i];

    // Rest is identical to the rotation variant — factor into shared device function
    // For now, duplicate the logic (will refactor if needed)

    int local_codes[kMaxDimsPerLane];

    if (num_bits == 0) {
        double partial_l2 = 0.0;
        for (size_t i = 0; i < my_dims; ++i)
            partial_l2 += (double)local_vec[i] * local_vec[i];
        double total_l2 = warp_reduce_sum_double(partial_l2);
        if (lane_id == 0) {
            d_o_l2norm[warp_id] = sqrtf((float)total_l2);
            d_fac_rescale[warp_id] = 0.0f;
            d_fac_error[warp_id] = 0.0f;
            d_ip_cent_oa[warp_id] = 0.0f;
        }
        return;
    }

    // v_max
    float local_max = 0.0f;
    for (size_t i = 0; i < my_dims; ++i)
        local_max = fmaxf(local_max, fabsf(local_vec[i]));
    float v_mx = warp_reduce_max(local_max);
    v_mx = warp_broadcast(v_mx);
    float v_mi = -v_mx;
    float delta = (v_mx - v_mi) / (code_max + 1);

    // Initial quantization
    double partial_ip_o_code = 0.0;
    uint64_t partial_code_l2sqr = 0;
    int partial_code_sum = 0;
    double partial_vec_sum = 0.0;
    double partial_o_l2sqr = 0.0;

    for (size_t i = 0; i < my_dims; ++i) {
        float o = local_vec[i];
        partial_o_l2sqr += (double)o * o;
        partial_vec_sum += o;
        int c;
        if (delta > 0.0f) {
            c = (int)floorf((o - v_mi) / delta);
            c = min(c, (int)code_max); c = max(c, 0);
        } else { c = 0; }
        local_codes[i] = c;
        partial_ip_o_code += (double)c * o;
        partial_code_l2sqr += (uint64_t)c * c;
        partial_code_sum += c;
    }

    double ip_o_code = warp_reduce_sum_double(partial_ip_o_code);
    double vec_sum = warp_reduce_sum_double(partial_vec_sum);
    double o_l2sqr = warp_reduce_sum_double(partial_o_l2sqr);
    double code_l2sqr_d = warp_reduce_sum_double((double)partial_code_l2sqr);
    double code_sum_d = warp_reduce_sum_double((double)partial_code_sum);
    ip_o_code = warp_broadcast_double(ip_o_code);
    vec_sum = warp_broadcast_double(vec_sum);
    o_l2sqr = warp_broadcast_double(o_l2sqr);
    code_l2sqr_d = warp_broadcast_double(code_l2sqr_d);
    code_sum_d = warp_broadcast_double(code_sum_d);

    double ip_o_oa = ip_o_code * delta + (v_mi + 0.5 * delta) * vec_sum;
    double oa_l2sqr = delta * delta * code_l2sqr_d
                    + (delta * delta + 2.0 * delta * v_mi) * code_sum_d
                    + (0.25 * delta * delta + delta * v_mi + (double)v_mi * v_mi) * D_seg;

    // Code adjustment
    if (caq_adj_rd_lmt && oa_l2sqr > 0.0 && delta > 0.0f) {
        double re_eps = (double)caq_adj_eps * oa_l2sqr;
        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int local_adj_cnt = 0;
            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                int c = local_codes[i];
                double oa = (c + 0.5) * delta + v_mi;
                double oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;
                while (c < (int)code_max) {
                    double new_q = oa + delta; double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa + ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr) break;
                    c++; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length; local_adj_cnt++;
                }
                while (c > 0) {
                    double new_q = oa - delta; double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr) break;
                    c--; ip_o_oa = new_ip; oa = new_q; oa_l2sqr = new_length; local_adj_cnt++;
                }
                local_codes[i] = c;
            }
            int total_adj = warp_reduce_sum_int(local_adj_cnt);
            total_adj = warp_broadcast_int(total_adj);
            if (total_adj == 0) break;
            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t i = 0; i < my_dims; ++i) {
                double q = (local_codes[i] + 0.5) * delta + v_mi;
                corr_ip += q * local_vec[i]; corr_oa_l2 += q * q;
            }
            oa_l2sqr = warp_broadcast_double(warp_reduce_sum_double(corr_oa_l2));
            ip_o_oa = warp_broadcast_double(warp_reduce_sum_double(corr_ip));
            re_eps = (double)caq_adj_eps * oa_l2sqr;
        }
    }

    // DownUpSample
    if (caq_ori_qB > 0) {
        int sampled_rshift = caq_ori_qB - (int)num_bits;
        delta *= (float)(1 << sampled_rshift);
        double new_ip = 0.0, new_oa_l2 = 0.0;
        for (size_t i = 0; i < my_dims; ++i) {
            local_codes[i] >>= sampled_rshift;
            double q = (local_codes[i] + 0.5) * delta + v_mi;
            new_ip += q * local_vec[i]; new_oa_l2 += q * q;
        }
        ip_o_oa = warp_broadcast_double(warp_reduce_sum_double(new_ip));
        oa_l2sqr = warp_broadcast_double(warp_reduce_sum_double(new_oa_l2));
    }

    // Factors
    double scale_rate = (v_mx > 0.0f) ? 1.0 / v_mx : 0.0;
    // rescale_vmx_to1: multiply by v_mx (matching CPU encoder)
    float fac_rescale_val = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa * v_mx) : 0.0f;
    float o_l2norm_val = sqrtf((float)o_l2sqr);
    constexpr float kConstEpsilon = 1.9f;
    float fac_error_val = 0.0f;
    if (ip_o_oa > 0.0 && D_seg > 1)
        fac_error_val = (float)(o_l2sqr * kConstEpsilon * sqrt(((o_l2sqr * oa_l2sqr) / (ip_o_oa * ip_o_oa) - 1.0) / (D_seg - 1)));

    float ip_c_oa = 0.0f;
    if (num_bits > 1) {
        double scaled_delta = delta * scale_rate;
        double scaled_vmi = v_mi * scale_rate;
        double partial_ip = 0.0;
        for (size_t i = 0; i < my_dims; ++i) {
            double oa_d = (local_codes[i] + 0.5) * scaled_delta + scaled_vmi;
            partial_ip += cent_base[start + i] * oa_d;
        }
        ip_c_oa = (float)warp_broadcast_double(warp_reduce_sum_double(partial_ip));
    }

    if (lane_id == 0) {
        d_o_l2norm[warp_id] = o_l2norm_val;
        d_fac_rescale[warp_id] = fac_rescale_val;
        d_fac_error[warp_id] = fac_error_val;
        d_ip_cent_oa[warp_id] = ip_c_oa;
    }

    // Pack short codes
    if (num_bits > 0 && d_short_raw) {
        size_t short_bytes_per_vec = D_seg / 8;
        for (size_t byte_idx = start / 8; byte_idx < (end + 7) / 8 && byte_idx < short_bytes_per_vec; ++byte_idx) {
            uint8_t byte_val = 0;
            for (int b = 0; b < 8; ++b) {
                size_t d = byte_idx * 8 + b;
                if (d >= start && d < end && d < D_seg) {
                    int c = local_codes[d - start];
                    byte_val |= ((c >> ((int)num_bits - 1)) & 1) << (7 - b);
                }
            }
            size_t byte_start_dim = byte_idx * 8;
            size_t byte_end_dim = byte_start_dim + 8;
            if (byte_start_dim >= start && byte_end_dim <= end) {
                d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = byte_val;
            } else {
                if (byte_start_dim == start || start == 0)
                    d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = 0;
                __syncwarp();
                if (byte_val != 0)
                    atomicOr((unsigned int*)(d_short_raw + (size_t)warp_id * short_bytes_per_vec + (byte_idx & ~3u)),
                             (unsigned int)byte_val << (8 * (byte_idx & 3u)));
            }
        }
    }

    // Pack long codes
    if (num_bits > 1 && d_long_raw) {
        size_t ex_bits = num_bits - 1;
        size_t long_bytes_per_vec = D_seg * ex_bits / 8;
        uint8_t* out = d_long_raw + (size_t)warp_id * long_bytes_per_vec;
        for (size_t i = 0; i < my_dims; ++i) {
            size_t d = start + i;
            int c = local_codes[i] & ((1 << ex_bits) - 1);
            size_t bit_offset = d * ex_bits;
            for (size_t b = 0; b < ex_bits; ++b) {
                size_t global_bit = bit_offset + b;
                size_t byte_pos = global_bit / 8;
                size_t bit_pos = global_bit % 8;
                if ((c >> b) & 1)
                    atomicOr((unsigned int*)(out + (byte_pos & ~3u)),
                             1u << (8 * (byte_pos & 3u) + bit_pos));
            }
        }
    }
}

// ============================================================================
// Launch wrappers
// ============================================================================

void launch_fused_caq_encode(
    const float* d_vectors_rotated,
    const float* d_rotated_centroids,
    const uint32_t* d_cluster_ids,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_short_raw, uint8_t* d_long_raw,
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    int caq_sequential,
    cudaStream_t stream)
{
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlockSize = 32 * kWarpsPerBlock;
    int grid = ((int)N + kWarpsPerBlock - 1) / kWarpsPerBlock;

    kernel_fused_caq_encode<<<grid, kBlockSize, 0, stream>>>(
        d_vectors_rotated, d_rotated_centroids, d_cluster_ids,
        d_o_l2norm, d_fac_rescale, d_fac_error, d_ip_cent_oa,
        d_short_raw, d_long_raw,
        D_seg, N, K, num_bits, code_max, caq_adj_rd_lmt, caq_adj_eps, caq_ori_qB,
        caq_sequential);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

void launch_fused_caq_encode_no_rotation(
    const float* d_vectors,
    const float* d_centroids,
    const uint32_t* d_cluster_ids,
    size_t seg_offset, size_t D_seg, size_t D_total,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_short_raw, uint8_t* d_long_raw,
    size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps, int caq_ori_qB,
    cudaStream_t stream)
{
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlockSize = 32 * kWarpsPerBlock;
    int grid = ((int)N + kWarpsPerBlock - 1) / kWarpsPerBlock;

    kernel_fused_caq_encode_no_rotation<<<grid, kBlockSize, 0, stream>>>(
        d_vectors, d_centroids, d_cluster_ids,
        seg_offset, D_seg, D_total,
        d_o_l2norm, d_fac_rescale, d_fac_error, d_ip_cent_oa,
        d_short_raw, d_long_raw,
        N, K, num_bits, code_max, caq_adj_rd_lmt, caq_adj_eps, caq_ori_qB);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Codebook Encode: binary search into sorted per-dimension centroids
// ============================================================================

/// Binary search in sorted codebook centroids for one dimension.
__device__ __forceinline__
int codebook_nearest(const float* codebook_dim, int num_entries, float value) {
    int lo = 0, hi = num_entries - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        float boundary = (__ldg(&codebook_dim[mid]) + __ldg(&codebook_dim[mid + 1])) * 0.5f;
        if (value <= boundary) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

__global__ void kernel_fused_codebook_encode(
    const float* __restrict__ d_vectors_rotated,
    const float* __restrict__ d_rotated_centroids,
    const uint32_t* __restrict__ d_cluster_ids,
    const float* __restrict__ d_codebook_centroids,  // [D_seg * entries_per_dim]
    size_t entries_per_dim,
    float* __restrict__ d_o_l2norm,
    float* __restrict__ d_fac_rescale,
    float* __restrict__ d_fac_error,
    float* __restrict__ d_ip_cent_oa,
    uint8_t* __restrict__ d_short_raw,
    uint8_t* __restrict__ d_long_raw,
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if ((size_t)warp_id >= N) return;

    uint32_t cid = d_cluster_ids[warp_id];

    size_t chunk = (D_seg + 31) / 32;
    size_t start = lane_id * chunk;
    size_t end = min(start + chunk, D_seg);
    size_t my_dims = (end > start) ? end - start : 0;

    const float* vec_rot = d_vectors_rotated + (size_t)warp_id * D_seg;
    const float* cent_rot = d_rotated_centroids + (size_t)cid * D_seg;

    float local_vec[kMaxDimsPerLane];
    int local_codes[kMaxDimsPerLane];

    // Compute residual
    for (size_t i = 0; i < my_dims; ++i) {
        local_vec[i] = vec_rot[start + i] - cent_rot[start + i];
    }

    if (num_bits == 0) {
        double partial_l2 = 0.0;
        for (size_t i = 0; i < my_dims; ++i)
            partial_l2 += (double)local_vec[i] * local_vec[i];
        double total_l2 = warp_reduce_sum_double(partial_l2);
        if (lane_id == 0) {
            d_o_l2norm[warp_id] = sqrtf((float)total_l2);
            d_fac_rescale[warp_id] = 0.0f;
            d_fac_error[warp_id] = 0.0f;
            d_ip_cent_oa[warp_id] = 0.0f;
        }
        return;
    }

    int nent = (int)entries_per_dim;

    // ---- Step 1: Binary search initial quantization ----
    double partial_o_l2sqr = 0.0;
    double partial_ip_o_oa = 0.0;
    double partial_oa_l2sqr = 0.0;

    for (size_t i = 0; i < my_dims; ++i) {
        float o = local_vec[i];
        partial_o_l2sqr += (double)o * o;

        size_t global_dim = start + i;
        const float* cb_dim = d_codebook_centroids + global_dim * entries_per_dim;
        int c = codebook_nearest(cb_dim, nent, o);
        local_codes[i] = c;

        float oa = __ldg(&cb_dim[c]);
        partial_ip_o_oa += (double)o * oa;
        partial_oa_l2sqr += (double)oa * oa;
    }

    double o_l2sqr = warp_reduce_sum_double(partial_o_l2sqr);
    double ip_o_oa = warp_reduce_sum_double(partial_ip_o_oa);
    double oa_l2sqr = warp_reduce_sum_double(partial_oa_l2sqr);
    o_l2sqr = warp_broadcast_double(o_l2sqr);
    ip_o_oa = warp_broadcast_double(ip_o_oa);
    oa_l2sqr = warp_broadcast_double(oa_l2sqr);

    // ---- Step 2: Codebook-aware code adjustment ----
    if (caq_adj_rd_lmt && oa_l2sqr > 0.0) {
        double re_eps = (double)caq_adj_eps * oa_l2sqr;

        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int local_adj_cnt = 0;

            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                int c = local_codes[i];
                size_t global_dim = start + i;
                const float* cb_dim = d_codebook_centroids + global_dim * entries_per_dim;
                float oa = __ldg(&cb_dim[c]);
                double oa_l2sqr_tmp = oa_l2sqr - (double)oa * oa;

                // Try increment
                while (c + 1 < nent) {
                    float new_oa = __ldg(&cb_dim[c + 1]);
                    double new_length = oa_l2sqr_tmp + (double)new_oa * new_oa;
                    double new_ip = ip_o_oa + (double)o * (new_oa - oa);
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = oa_l2sqr_tmp + (double)oa * oa;
                    local_adj_cnt++;
                }
                // Try decrement
                while (c > 0) {
                    float new_oa = __ldg(&cb_dim[c - 1]);
                    double new_length = oa_l2sqr_tmp + (double)new_oa * new_oa;
                    double new_ip = ip_o_oa + (double)o * (new_oa - oa);
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_o_oa = new_ip;
                    oa = new_oa;
                    oa_l2sqr = oa_l2sqr_tmp + (double)oa * oa;
                    local_adj_cnt++;
                }
                local_codes[i] = c;
            }

            int total_adj = warp_reduce_sum_int(local_adj_cnt);
            total_adj = warp_broadcast_int(total_adj);
            if (total_adj == 0) break;

            // Recompute global sums after adjustment round
            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t i = 0; i < my_dims; ++i) {
                float o = local_vec[i];
                size_t global_dim = start + i;
                float oa = __ldg(&d_codebook_centroids[global_dim * entries_per_dim + local_codes[i]]);
                corr_ip += (double)oa * o;
                corr_oa_l2 += (double)oa * oa;
            }
            oa_l2sqr = warp_reduce_sum_double(corr_oa_l2);
            ip_o_oa = warp_reduce_sum_double(corr_ip);
            oa_l2sqr = warp_broadcast_double(oa_l2sqr);
            ip_o_oa = warp_broadcast_double(ip_o_oa);
            re_eps = (double)caq_adj_eps * oa_l2sqr;
        }
    }

    // ---- Step 3: Compute factors (codebook mode) ----
    // Codebook rescale: o_l2sqr / ip_o_oa  (no v_mx factor)
    float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa) : 0.0f;
    float o_l2norm = sqrtf((float)o_l2sqr);

    constexpr float kConstEpsilon = 1.9f;
    float fac_error = 0.0f;
    if (ip_o_oa > 0.0 && D_seg > 1) {
        fac_error = (float)(o_l2sqr * kConstEpsilon *
            sqrt(((o_l2sqr * oa_l2sqr) / (ip_o_oa * ip_o_oa) - 1.0) / (D_seg - 1)));
    }

    // ---- Step 4: ip_cent_oa (centroid dot quantized) ----
    float ip_c_oa = 0.0f;
    {
        double partial_ip = 0.0;
        for (size_t i = 0; i < my_dims; ++i) {
            size_t global_dim = start + i;
            float oa = __ldg(&d_codebook_centroids[global_dim * entries_per_dim + local_codes[i]]);
            partial_ip += (double)cent_rot[start + i] * oa;
        }
        double total_ip = warp_reduce_sum_double(partial_ip);
        ip_c_oa = (float)warp_broadcast_double(total_ip);
    }

    // ---- Write scalar outputs (lane 0) ----
    if (lane_id == 0) {
        d_o_l2norm[warp_id] = o_l2norm;
        d_fac_rescale[warp_id] = fac_rescale;
        d_fac_error[warp_id] = fac_error;
        d_ip_cent_oa[warp_id] = ip_c_oa;
    }

    // ---- Step 5: Pack codes into short_raw and long_raw ----
    // Codebook mode: codes are full indices [0, entries_per_dim).
    // We pack them identically to the uniform path: 1-bit MSB into short_codes
    // (for fastscan compatibility), remaining bits into long_codes.

    // -- Short codes (1-bit MSB, descending bit order: dim 0 → bit 7) --
    if (num_bits > 0 && d_short_raw) {
        size_t short_bytes_per_vec = D_seg / 8;
        for (size_t byte_idx = start / 8; byte_idx < (end + 7) / 8 && byte_idx < short_bytes_per_vec; ++byte_idx) {
            uint8_t byte_val = 0;
            for (int b = 0; b < 8; ++b) {
                size_t d = byte_idx * 8 + b;
                if (d >= start && d < end && d < D_seg) {
                    int c = local_codes[d - start];
                    byte_val |= ((c >> ((int)num_bits - 1)) & 1) << (7 - b);
                }
            }
            size_t byte_start_dim = byte_idx * 8;
            size_t byte_end_dim = byte_start_dim + 8;
            if (byte_start_dim >= start && byte_end_dim <= end) {
                d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = byte_val;
            } else {
                if (byte_start_dim == start || start == 0)
                    d_short_raw[(size_t)warp_id * short_bytes_per_vec + byte_idx] = 0;
                __syncwarp();
                if (byte_val != 0)
                    atomicOr((unsigned int*)(d_short_raw + (size_t)warp_id * short_bytes_per_vec + (byte_idx & ~3u)),
                             (unsigned int)byte_val << (8 * (byte_idx & 3u)));
            }
        }
    }

    // -- Long codes (remaining bits, bit-compacted) --
    if (num_bits > 1 && d_long_raw) {
        size_t ex_bits = num_bits - 1;
        size_t long_bytes_per_vec = D_seg * ex_bits / 8;
        uint8_t* my_long = d_long_raw + (size_t)warp_id * long_bytes_per_vec;

        for (size_t i = 0; i < my_dims; ++i) {
            size_t d = start + i;
            int code_low = local_codes[i] & ((1 << ex_bits) - 1);
            size_t bit_offset = d * ex_bits;
            for (size_t b = 0; b < ex_bits; ++b) {
                if ((code_low >> b) & 1) {
                    size_t gbit = bit_offset + b;
                    size_t byte_pos = gbit / 8;
                    size_t bit_pos = gbit % 8;
                    atomicOr((unsigned int*)(my_long + (byte_pos & ~3u)),
                             1u << (8 * (byte_pos & 3u) + bit_pos));
                }
            }
        }
    }
}

void launch_fused_codebook_encode(
    const float* d_vectors_rotated,
    const float* d_rotated_centroids,
    const uint32_t* d_cluster_ids,
    const float* d_codebook_centroids,
    size_t entries_per_dim,
    float* d_o_l2norm, float* d_fac_rescale,
    float* d_fac_error, float* d_ip_cent_oa,
    uint8_t* d_short_raw, uint8_t* d_long_raw,
    size_t D_seg, size_t N, size_t K,
    size_t num_bits, uint16_t code_max,
    int caq_adj_rd_lmt, float caq_adj_eps,
    cudaStream_t stream)
{
    if (N == 0) return;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlockSize = 32 * kWarpsPerBlock;
    int grid = ((int)N + kWarpsPerBlock - 1) / kWarpsPerBlock;

    kernel_fused_codebook_encode<<<grid, kBlockSize, 0, stream>>>(
        d_vectors_rotated, d_rotated_centroids, d_cluster_ids,
        d_codebook_centroids, entries_per_dim,
        d_o_l2norm, d_fac_rescale, d_fac_error, d_ip_cent_oa,
        d_short_raw, d_long_raw,
        D_seg, N, K, num_bits, code_max,
        caq_adj_rd_lmt, caq_adj_eps);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
