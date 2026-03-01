#include "saq/gpu/gpu_encoder.cuh"
#include "saq/gpu/gpu_utils.cuh"

namespace saq::gpu {

__global__ void kernel_subtract_centroid(
    const float* __restrict__ d_vectors,
    const float* __restrict__ d_centroids,
    const uint32_t* __restrict__ d_cluster_ids,
    float* __restrict__ d_residuals,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    size_t N) {

    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= N) return;

    uint32_t cid = d_cluster_ids[vec_idx];
    const float* vec = d_vectors + vec_idx * D_total + seg_offset;
    const float* cen = d_centroids + cid * D_total + seg_offset;
    float* out = d_residuals + vec_idx * D_seg;

    for (size_t d = 0; d < D_seg; ++d) {
        out[d] = vec[d] - cen[d];
    }
}

void launch_subtract_centroid(
    const float* d_vectors,
    const float* d_centroids,
    const uint32_t* d_cluster_ids,
    float* d_residuals,
    size_t seg_offset,
    size_t D_seg,
    size_t D_total,
    size_t N,
    cudaStream_t stream) {

    constexpr int kBlockSize = 256;
    int grid = (N + kBlockSize - 1) / kBlockSize;
    kernel_subtract_centroid<<<grid, kBlockSize, 0, stream>>>(
        d_vectors, d_centroids, d_cluster_ids, d_residuals,
        seg_offset, D_seg, D_total, N);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Warp-cooperative CAQ Encode
// ============================================================================

// Each warp (32 threads) processes one vector cooperatively.
// Lane i handles dimensions [i*chunk, (i+1)*chunk) where chunk = ceil(D_seg/32).

__global__ void kernel_caq_encode(
    const float* __restrict__ d_rotated,  // [N x D_seg]
    int* __restrict__ d_codes,            // [N x D_seg]
    float* __restrict__ d_o_l2norm,       // [N]
    float* __restrict__ d_fac_rescale,    // [N]
    float* __restrict__ d_fac_error,      // [N]
    float* __restrict__ d_ip_cent_oa,     // [N]
    const float* __restrict__ d_centroids_seg, // [K x D_seg]
    const uint32_t* __restrict__ d_cluster_ids, // [N]
    size_t D_seg,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB) {

    // One warp per vector
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if ((size_t)warp_id >= N) return;
    if (num_bits == 0) {
        // Zero-bit segment: just compute o_l2norm
        const float* vec = d_rotated + (size_t)warp_id * D_seg;
        size_t chunk = (D_seg + 31) / 32;
        size_t start = lane_id * chunk;
        size_t end = min(start + chunk, D_seg);
        double partial_l2 = 0.0;
        for (size_t d = start; d < end; ++d)
            partial_l2 += (double)vec[d] * vec[d];
        double total_l2 = warp_reduce_sum_double(partial_l2);
        if (lane_id == 0) {
            d_o_l2norm[warp_id] = sqrtf((float)total_l2);
            d_fac_rescale[warp_id] = 0.0f;
            d_fac_error[warp_id] = 0.0f;
            d_ip_cent_oa[warp_id] = 0.0f;
        }
        return;
    }

    const float* vec = d_rotated + (size_t)warp_id * D_seg;
    int* codes = d_codes + (size_t)warp_id * D_seg;

    // Each lane handles a chunk of dimensions
    size_t chunk = (D_seg + 31) / 32;
    size_t start = lane_id * chunk;
    size_t end = min(start + chunk, D_seg);

    // ---- Step 1: Compute v_max ----
    float local_max = 0.0f;
    for (size_t d = start; d < end; ++d) {
        float v = vec[d];
        local_max = fmaxf(local_max, fabsf(v));
    }
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

    for (size_t d = start; d < end; ++d) {
        float o = vec[d];
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
        codes[d] = c;
        partial_ip_o_code += (double)c * o;
        partial_code_l2sqr += (uint64_t)c * c;
        partial_code_sum += c;
    }

    // Warp reduce to get global sums
    double ip_o_code = warp_reduce_sum_double(partial_ip_o_code);
    double vec_sum = warp_reduce_sum_double(partial_vec_sum);
    double o_l2sqr = warp_reduce_sum_double(partial_o_l2sqr);

    // For code_l2sqr and code_sum, reduce as doubles to avoid overflow
    double code_l2sqr_d = warp_reduce_sum_double((double)partial_code_l2sqr);
    double code_sum_d = warp_reduce_sum_double((double)partial_code_sum);

    // Broadcast to all lanes
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

        for (int round = 1; round <= caq_adj_rd_lmt || caq_adj_rd_lmt == 0; ++round) {
            int local_adj_cnt = 0;

            for (size_t d = start; d < end; ++d) {
                float o = vec[d];
                int c = codes[d];
                double oa = (c + 0.5) * delta + v_mi;
                double oa_l2sqr_tmp = oa_l2sqr - oa * oa;
                double ip_delta = delta * o;

                // Try ++
                while (c < (int)code_max) {
                    double new_q = oa + delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa + ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c++;
                    ip_o_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                // Try --
                while (c > 0) {
                    double new_q = oa - delta;
                    double new_length = oa_l2sqr_tmp + new_q * new_q;
                    double new_ip = ip_o_oa - ip_delta;
                    if ((ip_o_oa * ip_o_oa + re_eps) * new_length >= new_ip * new_ip * oa_l2sqr)
                        break;
                    c--;
                    ip_o_oa = new_ip;
                    oa = new_q;
                    oa_l2sqr = new_length;
                    local_adj_cnt++;
                }
                codes[d] = c;
            }

            // Check convergence across warp
            int total_adj = warp_reduce_sum_int(local_adj_cnt);
            total_adj = warp_broadcast_int(total_adj);
            if (total_adj == 0) break;

            // Correction pass: recompute sums from codes
            double corr_oa_l2 = 0.0, corr_ip = 0.0;
            for (size_t d = start; d < end; ++d) {
                float o = vec[d];
                double q = (codes[d] + 0.5) * delta + v_mi;
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

    // ---- Step 3b: DownUpSample (if caq_ori_qB > 0) ----
    if (caq_ori_qB > 0) {
        int sampled_rshift = caq_ori_qB - (int)num_bits;
        delta *= (float)(1 << sampled_rshift);
        // Reset and recompute
        double new_ip = 0.0, new_oa_l2 = 0.0;
        for (size_t d = start; d < end; ++d) {
            codes[d] >>= sampled_rshift;
            float o = vec[d];
            double q = (codes[d] + 0.5) * delta + v_mi;
            new_ip += q * o;
            new_oa_l2 += q * q;
        }
        ip_o_oa = warp_reduce_sum_double(new_ip);
        oa_l2sqr = warp_reduce_sum_double(new_oa_l2);
        ip_o_oa = warp_broadcast_double(ip_o_oa);
        oa_l2sqr = warp_broadcast_double(oa_l2sqr);
    }

    // ---- Step 4: Compute factors ----
    // rescale_vmx_to1: scale so v_mx = 1
    double scale_rate = (v_mx > 0.0f) ? 1.0 / v_mx : 0.0;
    double scaled_ip = ip_o_oa * scale_rate;
    double scaled_oa_l2 = oa_l2sqr * scale_rate * scale_rate;

    float fac_rescale = (ip_o_oa != 0.0) ? (float)(o_l2sqr / ip_o_oa) : 0.0f;
    float o_l2norm = sqrtf((float)o_l2sqr);

    constexpr float kConstEpsilon = 1.9f;
    float fac_error = 0.0f;
    if (ip_o_oa > 0.0 && D_seg > 1) {
        fac_error = (float)(o_l2sqr * kConstEpsilon *
            sqrt(((o_l2sqr * oa_l2sqr) / (ip_o_oa * ip_o_oa) - 1.0) / (D_seg - 1)));
    }

    // ---- Step 5: ip_cent_oa (if num_bits > 1) ----
    float ip_c_oa = 0.0f;
    if (num_bits > 1) {
        uint32_t cid = d_cluster_ids[warp_id];
        const float* cent = d_centroids_seg + (size_t)cid * D_seg;
        // After rescale_vmx_to1: oa = (code + 0.5) * (delta * scale_rate) + (v_mi * scale_rate)
        double scaled_delta = delta * scale_rate;
        double scaled_vmi = v_mi * scale_rate;
        double partial_ip = 0.0;
        for (size_t d = start; d < end; ++d) {
            double oa_d = (codes[d] + 0.5) * scaled_delta + scaled_vmi;
            partial_ip += cent[d] * oa_d;
        }
        double total_ip = warp_reduce_sum_double(partial_ip);
        ip_c_oa = (float)warp_broadcast_double(total_ip);
    }

    // ---- Write outputs (lane 0 only for scalars) ----
    if (lane_id == 0) {
        d_o_l2norm[warp_id] = o_l2norm;
        d_fac_rescale[warp_id] = fac_rescale;
        d_fac_error[warp_id] = fac_error;
        d_ip_cent_oa[warp_id] = ip_c_oa;
    }
    // All lanes write their chunk of codes (already written in-place above)
}

void launch_caq_encode(
    const float* d_rotated,
    int* d_codes,
    float* d_o_l2norm,
    float* d_fac_rescale,
    float* d_fac_error,
    float* d_ip_cent_oa,
    const float* d_centroids_seg,
    const uint32_t* d_cluster_ids,
    size_t D_seg,
    size_t N,
    size_t K,
    size_t num_bits,
    uint16_t code_max,
    int caq_adj_rd_lmt,
    float caq_adj_eps,
    int caq_ori_qB,
    cudaStream_t stream) {

    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlockSize = 32 * kWarpsPerBlock;
    int grid = ((int)N + kWarpsPerBlock - 1) / kWarpsPerBlock;

    kernel_caq_encode<<<grid, kBlockSize, 0, stream>>>(
        d_rotated, d_codes, d_o_l2norm, d_fac_rescale, d_fac_error, d_ip_cent_oa,
        d_centroids_seg, d_cluster_ids,
        D_seg, N, K, num_bits, code_max, caq_adj_rd_lmt, caq_adj_eps, caq_ori_qB);
    SAQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace saq::gpu
