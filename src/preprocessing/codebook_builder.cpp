#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include <glog/logging.h>

namespace saq {

namespace {

struct Prefix {           // per-point prefix sums over a sorted column
    std::vector<double> ps, psq;  // size n+1
    size_t n = 0;
    double count(size_t a, size_t b) const { return double(b - a + 1); }       // [a,b] inclusive
    double sum(size_t a, size_t b)   const { return ps[b + 1] - ps[a]; }
    double mean(size_t a, size_t b)  const { return sum(a, b) / count(a, b); }
    double sse(size_t a, size_t b)   const {
        double c = count(a, b), s = sum(a, b);
        return (psq[b + 1] - psq[a]) - s * s / c;
    }
};

Prefix make_prefix(const std::vector<float>& s) {
    Prefix p; p.n = s.size();
    p.ps.assign(p.n + 1, 0.0); p.psq.assign(p.n + 1, 0.0);
    for (size_t i = 0; i < p.n; ++i) {
        double v = s[i]; p.ps[i + 1] = p.ps[i] + v; p.psq[i + 1] = p.psq[i] + v * v;
    }
    return p;
}

size_t num_distinct(const std::vector<float>& s) {
    if (s.empty()) return 0;
    size_t d = 1; for (size_t i = 1; i < s.size(); ++i) if (s[i] != s[i - 1]) ++d;
    return d;
}

// Make centroids strictly increasing using a magnitude-relative epsilon, so the
// gap survives the later narrowing to float even at large value magnitudes
// (float ULP ~ |x|*1.2e-7; |x|*1e-6 stays above it).
void nudge_strictly_increasing(std::vector<double>& c) {
    for (size_t i = 1; i < c.size(); ++i)
        if (c[i] <= c[i - 1])
            c[i] = c[i - 1] + std::max(1e-9, std::fabs(c[i - 1]) * 1e-6);
}

// Equal-mass quantile init: k centroids = means of k equal-count cells.
std::vector<double> init_equal_mass(const Prefix& pf, size_t n, size_t k) {
    std::vector<double> c(k);
    for (size_t i = 0; i < k; ++i) {
        size_t a = (i * n) / k, b = ((i + 1) * n) / k;
        if (b <= a) b = a + 1; if (b > n) b = n;
        c[i] = pf.mean(a, b - 1);
    }
    nudge_strictly_increasing(c);
    return c;
}

// One Lloyd run for fixed k. Mutates c (sorted asc), returns total SSE.
double lloyd_k(const std::vector<float>& s, const Prefix& pf, size_t k,
               std::vector<double>& c, size_t max_iters, float tol) {
    const size_t n = s.size();
    std::vector<size_t> bnd(k + 1);
    auto assign = [&]() {
        bnd[0] = 0; bnd[k] = n;
        for (size_t i = 1; i < k; ++i) {
            float mid = static_cast<float>(0.5 * (c[i - 1] + c[i]));
            bnd[i] = static_cast<size_t>(
                std::upper_bound(s.begin(), s.end(), mid) - s.begin());
        }
        for (size_t i = 1; i < k; ++i) if (bnd[i] < bnd[i - 1]) bnd[i] = bnd[i - 1];
    };
    size_t repairs_done = 0;
    const size_t repair_budget = 2 * k;  // cap so pathological oscillation can't starve convergence
    for (size_t iter = 0; iter < max_iters; ++iter) {
        assign();
        double maxmove = 0.0;
        for (size_t i = 0; i < k; ++i) {
            size_t a = bnd[i], b = bnd[i + 1];
            double nc = (b > a) ? pf.mean(a, b - 1) : c[i];
            maxmove = std::max(maxmove, std::fabs(nc - c[i])); c[i] = nc;
        }
        // Empty-cell repair: at most ONE split per iteration. Repairing every
        // empty cell at once against the stale bnd[] lets two empties target the
        // same donor and clobber each other; doing one split then reassigning
        // next iteration avoids that. Bounded by repair_budget so a pathological
        // oscillation cannot starve the convergence check.
        bool repaired = false;
        if (repairs_done < repair_budget) {
            size_t empty_i = k;
            for (size_t i = 0; i < k; ++i) if (bnd[i + 1] <= bnd[i]) { empty_i = i; break; }
            if (empty_i < k) {
                size_t best = k; double bsse = -1.0;
                for (size_t j = 0; j < k; ++j) {
                    size_t a = bnd[j], b = bnd[j + 1];
                    if (b > a + 1) { double e = pf.sse(a, b - 1); if (e > bsse) { bsse = e; best = j; } }
                }
                if (best < k) {  // a splittable donor exists (guaranteed when k < ndist)
                    size_t a = bnd[best], b = bnd[best + 1];
                    size_t m = a + (b - a) / 2;
                    c[best]    = pf.mean(a, m - 1);  // donor lower half
                    c[empty_i] = pf.mean(m, b - 1);  // empty slot takes upper half
                    std::sort(c.begin(), c.end());
                    nudge_strictly_increasing(c);
                    ++repairs_done;
                    repaired = true;
                }
            }
        }
        if (repaired) continue;   // reassign with repaired centroids next iteration
        if (maxmove < tol) break;
    }
    assign();
    double sse = 0.0;
    for (size_t i = 0; i < k; ++i) { size_t a = bnd[i], b = bnd[i + 1]; if (b > a) sse += pf.sse(a, b - 1); }
    return sse;
}

// Init dispatch (Task 4 extends with Uniform/KMeans++).
std::vector<double> init_centroids(const std::vector<float>& s, const Prefix& pf,
                                   size_t k, const LloydOpts& opts, size_t /*restart*/) {
    (void)s; (void)opts;
    return init_equal_mass(pf, pf.n, k);
}

}  // namespace

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
    // C = per-bin count, S = per-bin sum, Q = per-bin sum-of-squares;
    // pc/ps/pq = their inclusive prefix sums (length B+1, index 0 is sentinel 0).
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
        // INF sentinel: summed double-precision SSE stays well below 1e30 for
        // the bin counts (<=500) and value ranges used in practice.
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
        // Traceback: runs exactly k iterations (one per cluster j), bounded by j
        // not by i; split[0] defaults to 0 so the leftmost cluster starts at bin 0.
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
CodebookResult build_codebook_lloyd(std::span<const float> values, const LloydOpts& opts) {
    CodebookResult R;
    R.costs.assign(opts.max_bits + 1, 0.f);
    R.codebooks.assign(opts.max_bits + 1, {});
    const size_t n = values.size();
    if (n == 0) return R;

    std::vector<float> s(values.begin(), values.end());
    std::sort(s.begin(), s.end());
    Prefix pf = make_prefix(s);
    const size_t ndist = num_distinct(s);

    R.costs[0] = static_cast<float>(pf.sse(0, n - 1) / n);
    R.codebooks[0].centroids = { static_cast<float>(pf.mean(0, n - 1)) };
    R.codebooks[0].num_entries = 1;

    for (size_t bits = 1; bits <= opts.max_bits; ++bits) {
        const size_t k = size_t(1) << bits;
        if (k >= ndist) {  // degenerate: every distinct value is its own centroid
            std::vector<float> cen;
            for (size_t i = 0; i < n; ++i) if (i == 0 || s[i] != s[i - 1]) cen.push_back(s[i]);
            R.codebooks[bits].centroids = cen;
            R.codebooks[bits].num_entries = cen.size();
            R.costs[bits] = 0.f;
            continue;
        }
        double best_sse = std::numeric_limits<double>::infinity();
        std::vector<double> best_c;
        const size_t restarts = std::max<size_t>(1, opts.restarts);
        for (size_t r = 0; r < restarts; ++r) {
            std::vector<double> c = init_centroids(s, pf, k, opts, r);
            double sse = lloyd_k(s, pf, k, c, opts.max_iters, opts.tol);
            if (sse < best_sse) { best_sse = sse; best_c = c; }
        }
        std::vector<float> cen(best_c.begin(), best_c.end());
        std::sort(cen.begin(), cen.end());
        // Drop any centroids that coincide after narrowing to float so num_entries
        // honestly reflects the distinct codebook size.
        cen.erase(std::unique(cen.begin(), cen.end()), cen.end());
        R.codebooks[bits].centroids = cen;
        R.codebooks[bits].num_entries = cen.size();
        R.costs[bits] = static_cast<float>(best_sse / n);
    }
    return R;
}
std::vector<CodebookResult> build_all_dims(const FloatRowMat&, const LloydOpts&) {
    return {};  // Task 5
}

}  // namespace saq
