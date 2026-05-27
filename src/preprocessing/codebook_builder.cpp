#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include <glog/logging.h>

namespace saq {

CodebookResult build_codebook_dp(std::span<const float> values,
                                 size_t max_bits, size_t num_bins) {
    CHECK_LE(max_bits, 8u) << "DP reference only valid for <= 8 bits";
    CodebookResult R;
    R.costs.assign(max_bits + 1, 0.f);
    R.codebooks.assign(max_bits + 1, {});
    const size_t n = values.size();
    if (n == 0) return R;

    std::vector<double> a(values.begin(), values.end());
    std::sort(a.begin(), a.end());
    num_bins = std::min(num_bins, n);

    const double lo = a.front() - 1e-10, hi = a.back() + 1e-10;
    const double width = (hi - lo) / static_cast<double>(num_bins);

    std::vector<double> bc(num_bins, 0), bs(num_bins, 0), bsq(num_bins, 0);
    for (double v : a) {
        size_t bi = width > 0 ? std::min(num_bins - 1,
                       static_cast<size_t>((v - lo) / width)) : 0;
        bc[bi] += 1; bs[bi] += v; bsq[bi] += v * v;
    }
    std::vector<double> C, S, Q;
    for (size_t i = 0; i < num_bins; ++i)
        if (bc[i] > 0.5) { C.push_back(bc[i]); S.push_back(bs[i]); Q.push_back(bsq[i]); }
    const size_t B = C.size();

    std::vector<double> pc(B + 1, 0), ps(B + 1, 0), pq(B + 1, 0);
    for (size_t i = 0; i < B; ++i) {
        pc[i + 1] = pc[i] + C[i]; ps[i + 1] = ps[i] + S[i]; pq[i + 1] = pq[i] + Q[i];
    }
    auto rsse = [&](size_t x, size_t y) {
        double c = pc[y + 1] - pc[x]; if (c < 0.5) return 0.0;
        double s = ps[y + 1] - ps[x], q = pq[y + 1] - pq[x]; return q - s * s / c;
    };
    auto rcen = [&](size_t x, size_t y) {
        double c = pc[y + 1] - pc[x]; if (c < 0.5) return 0.0;
        return (ps[y + 1] - ps[x]) / c;
    };

    R.costs[0] = static_cast<float>(rsse(0, B - 1) / n);
    R.codebooks[0].centroids = { static_cast<float>(rcen(0, B - 1)) };
    R.codebooks[0].num_entries = 1;

    for (size_t bits = 1; bits <= max_bits; ++bits) {
        const size_t k = size_t(1) << bits;
        if (k >= B) {  // more clusters than bins: each bin its own centroid
            double tot = 0; std::vector<float> cen;
            for (size_t i = 0; i < B; ++i) {
                tot += Q[i] - S[i] * S[i] / C[i];
                cen.push_back(static_cast<float>(S[i] / C[i]));
            }
            R.costs[bits] = static_cast<float>(tot / n);
            R.codebooks[bits].centroids = cen;
            R.codebooks[bits].num_entries = cen.size();
            continue;
        }
        const double INF = 1e30;
        std::vector<double> prev(B), cur(B);
        std::vector<std::vector<int>> split(k, std::vector<int>(B, 0));
        for (size_t i = 0; i < B; ++i) prev[i] = rsse(0, i);  // j=1 base
        for (size_t j = 2; j <= k; ++j) {
            std::fill(cur.begin(), cur.end(), INF);
            for (size_t i = j - 1; i < B; ++i) {
                double best = INF; int bm = static_cast<int>(j - 1);
                for (size_t m = j - 1; m <= i; ++m) {
                    double pcst = (m > 0) ? prev[m - 1] : 0.0;
                    double tot = pcst + rsse(m, i);
                    if (tot < best) { best = tot; bm = static_cast<int>(m); }
                }
                cur[i] = best; split[j - 1][i] = bm;
            }
            prev = cur;
        }
        R.costs[bits] = static_cast<float>(prev[B - 1] / n);
        std::vector<float> cen; int i = static_cast<int>(B) - 1;
        for (int j = static_cast<int>(k) - 1; j >= 0; --j) {
            int m = split[j][i]; cen.push_back(static_cast<float>(rcen(m, i))); i = m - 1;
        }
        std::reverse(cen.begin(), cen.end());
        R.codebooks[bits].centroids = cen;
        R.codebooks[bits].num_entries = cen.size();
    }
    return R;
}
CodebookResult build_codebook_lloyd(std::span<const float>, const LloydOpts&) {
    return {};  // Task 2
}
std::vector<CodebookResult> build_all_dims(const FloatRowMat&, const LloydOpts&) {
    return {};  // Task 5
}

}  // namespace saq
