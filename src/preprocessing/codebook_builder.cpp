#include "saq/preprocessing/codebook_builder.h"

#include <algorithm>
#include <cmath>
#include <functional>
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

// Uniform-spaced init: k centroids evenly spaced over [min, max].
std::vector<double> init_uniform(const std::vector<float>& s, size_t k) {
    std::vector<double> c(k);
    const double lo = s.front(), hi = s.back();
    if (hi <= lo) {                       // degenerate: all values equal
        std::fill(c.begin(), c.end(), lo);
    } else {
        for (size_t i = 0; i < k; ++i)
            c[i] = lo + (hi - lo) * (static_cast<double>(i) + 0.5) / static_cast<double>(k);
    }
    nudge_strictly_increasing(c);
    return c;
}

// k-means++ (D^2-weighted) seeding over the sorted values.
// D^2(x) = squared distance to nearest already-chosen center; next center
// sampled with probability proportional to D^2.
std::vector<double> init_kmeanspp(const std::vector<float>& s, size_t k, uint64_t seed) {
    const size_t n = s.size();
    std::mt19937_64 rng(seed);
    std::vector<double> centers;
    centers.reserve(k);
    // First center: uniform random point.
    centers.push_back(s[std::uniform_int_distribution<size_t>(0, n - 1)(rng)]);
    // Nearest-center squared distance for every point.
    std::vector<double> d2(n);
    for (size_t i = 0; i < n; ++i) {
        double diff = double(s[i]) - centers[0];
        d2[i] = diff * diff;
    }
    while (centers.size() < k) {
        double total = 0.0;
        for (double v : d2) total += v;
        size_t chosen;
        if (total <= 0.0) {  // all remaining points coincide with centers
            chosen = std::uniform_int_distribution<size_t>(0, n - 1)(rng);
        } else {
            double target = std::uniform_real_distribution<double>(0.0, total)(rng);
            double acc = 0.0;
            chosen = n - 1;
            for (size_t i = 0; i < n; ++i) {
                acc += d2[i];
                if (acc >= target) { chosen = i; break; }
            }
        }
        double cv = s[chosen];
        centers.push_back(cv);
        // Update nearest-center distances.
        for (size_t i = 0; i < n; ++i) {
            double diff = double(s[i]) - cv;
            double dd = diff * diff;
            if (dd < d2[i]) d2[i] = dd;
        }
    }
    std::sort(centers.begin(), centers.end());
    nudge_strictly_increasing(centers);
    return centers;
}

// Cube-root-density (companding) init. Optimal scalar-quantizer point density
// is proportional to f(x)^(1/3); we approximate f via a histogram and place
// centroids at equal increments of the cube-root-density CDF.
std::vector<double> init_cuberoot(const std::vector<float>& s, size_t k) {
    std::vector<double> c(k);
    const double lo = s.front(), hi = s.back();
    if (hi <= lo) {                       // degenerate: all values equal
        std::fill(c.begin(), c.end(), lo);
        nudge_strictly_increasing(c);
        return c;
    }
    const size_t B = 1000;
    const double width = (hi - lo) / static_cast<double>(B);
    std::vector<double> count(B, 0.0);
    for (float x : s) {
        size_t b = static_cast<size_t>((double(x) - lo) / width);
        if (b >= B) b = B - 1;            // hi maps to last bin
        count[b] += 1.0;
    }
    // g[b] = pow(count/width, 1/3) * width = pow(count,1/3) * pow(width,2/3)
    // cumulative G[b].
    std::vector<double> G(B);
    double acc = 0.0;
    for (size_t b = 0; b < B; ++b) {
        double g = (count[b] > 0.0) ? std::cbrt(count[b]) * std::pow(width, 2.0 / 3.0) : 0.0;
        acc += g;
        G[b] = acc;
    }
    const double Gtot = G[B - 1];
    for (size_t i = 0; i < k; ++i) {
        double t = (static_cast<double>(i) + 0.5) / static_cast<double>(k) * Gtot;
        // First bin whose cumulative G crosses t.
        size_t b = static_cast<size_t>(
            std::lower_bound(G.begin(), G.end(), t) - G.begin());
        if (b >= B) b = B - 1;
        // Interpolate within the bin using the fraction of this bin's mass
        // needed to reach t.
        double Gprev = (b > 0) ? G[b - 1] : 0.0;
        double gbin = G[b] - Gprev;
        double frac = (gbin > 0.0) ? (t - Gprev) / gbin : 0.5;
        if (frac < 0.0) frac = 0.0; if (frac > 1.0) frac = 1.0;
        c[i] = lo + (static_cast<double>(b) + frac) * width;
    }
    nudge_strictly_increasing(c);
    return c;
}

// Init dispatch honoring opts.init and the restart index.
std::vector<double> init_centroids(const std::vector<float>& s, const Prefix& pf,
                                   size_t k, const LloydOpts& opts, size_t restart) {
    switch (opts.init) {
        case CodebookInit::UniformSpaced:
            return init_uniform(s, k);
        case CodebookInit::KMeansPlusPlus:
            return init_kmeanspp(s, k, opts.seed + restart);
        case CodebookInit::CubeRootDensity:
            return init_cuberoot(s, k);
        case CodebookInit::EqualMassQuantile:
        default:
            return init_equal_mass(pf, pf.n, k);
    }
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

    // Build the working set: full data, or a deterministic random sample.
    std::vector<float> s;
    if (opts.sample_size > 0 && opts.sample_size < n) {
        const size_t m = opts.sample_size;
        s.resize(m);
        std::mt19937_64 rng(opts.seed);
        // Sample m indices without replacement (partial Fisher-Yates over an
        // index permutation would be O(n); reservoir sampling keeps it O(n)
        // and deterministic given the seed).
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        for (size_t i = 0; i < m; ++i) {
            size_t j = i + std::uniform_int_distribution<size_t>(0, n - 1 - i)(rng);
            std::swap(idx[i], idx[j]);
            s[i] = values[idx[i]];
        }
    } else {
        s.assign(values.begin(), values.end());
    }
    const size_t ns = s.size();
    std::sort(s.begin(), s.end());
    Prefix pf = make_prefix(s);
    const size_t ndist = num_distinct(s);

    R.costs[0] = static_cast<float>(pf.sse(0, ns - 1) / ns);
    R.codebooks[0].centroids = { static_cast<float>(pf.mean(0, ns - 1)) };
    R.codebooks[0].num_entries = 1;

    for (size_t bits = 1; bits <= opts.max_bits; ++bits) {
        const size_t k = size_t(1) << bits;
        if (k >= ndist) {  // degenerate: every distinct value is its own centroid
            std::vector<float> cen;
            for (size_t i = 0; i < ns; ++i) if (i == 0 || s[i] != s[i - 1]) cen.push_back(s[i]);
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
        // Drop any centroids that coincide after narrowing to float.
        cen.erase(std::unique(cen.begin(), cen.end()), cen.end());
        // Pad back up to 2^bits so per-dim codebooks in the same segment are
        // uniform in size — required by the GPU codebook upload, which reads
        // a single num_entries per segment. Padding with the last centroid is
        // a no-op for nearest() (duplicates collapse in binary search). The
        // resulting codebook is non-decreasing with strict increase among the
        // unique prefix; trailing duplicates exist only when dedup removed entries.
        const size_t k_target = size_t(1) << bits;
        while (cen.size() < k_target) cen.push_back(cen.back());
        R.codebooks[bits].centroids = cen;
        R.codebooks[bits].num_entries = cen.size();
        R.costs[bits] = static_cast<float>(best_sse / ns);
    }
    return R;
}
CodebookResult build_codebook_exact(std::span<const float> values, size_t max_bits) {
    CodebookResult R;
    R.costs.assign(max_bits + 1, 0.f);
    R.codebooks.assign(max_bits + 1, {});
    const size_t n = values.size();
    if (n == 0) return R;

    std::vector<double> a(values.begin(), values.end());
    std::sort(a.begin(), a.end());
    size_t ndist = 1;
    for (size_t i = 1; i < n; ++i) if (a[i] != a[i - 1]) ++ndist;

    // Prefix sums over sorted raw values; SSE/centroid of half-open range [i, j).
    std::vector<double> ps(n + 1, 0.0), pq(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) { ps[i + 1] = ps[i] + a[i]; pq[i + 1] = pq[i] + a[i] * a[i]; }
    auto sse = [&](size_t i, size_t j) -> double {
        if (j <= i) return 0.0;
        double c = double(j - i), s = ps[j] - ps[i];
        return (pq[j] - pq[i]) - s * s / c;
    };
    auto cen = [&](size_t i, size_t j) -> double { return (ps[j] - ps[i]) / double(j - i); };

    // bits = 0 : single centroid (global mean).
    R.costs[0] = static_cast<float>(sse(0, n) / n);
    R.codebooks[0].centroids = { static_cast<float>(cen(0, n)) };
    R.codebooks[0].num_entries = 1;

    const size_t K = size_t(1) << max_bits;
    const double INF = std::numeric_limits<double>::infinity();
    const size_t Klev = std::min(K, ndist);              // levels we actually run the DP for

    // D_prev/D_cur[m] = optimal SSE of the first m sorted points in (k'-1)/k' clusters.
    // A[k'][m] = optimal split = start index of the last (k'-th) cluster. Stored for
    // every level so each recorded bit-rate can be backtracked.
    std::vector<double> D_prev(n + 1, INF), D_cur(n + 1, INF);
    std::vector<int> A((Klev + 1) * (n + 1), 0);
    auto Aat = [&](size_t kp, size_t m) -> int& { return A[kp * (n + 1) + m]; };
    for (size_t m = 1; m <= n; ++m) { D_prev[m] = sse(0, m); Aat(1, m) = 0; }

    // Divide-and-conquer over m for one DP level kp: the optimal split is monotone in
    // m, so total work for the level is O(n log n).
    std::function<void(size_t, size_t, size_t, size_t, size_t)> dc =
        [&](size_t kp, size_t mlo, size_t mhi, size_t jlo, size_t jhi) {
            if (mlo > mhi) return;
            size_t mid = (mlo + mhi) / 2;
            double best = INF; size_t arg = jlo;
            size_t lo = std::max(kp - 1, jlo), hi = std::min(mid - 1, jhi);
            for (size_t j = lo; j <= hi; ++j) {
                double v = D_prev[j] + sse(j, mid);
                if (v < best) { best = v; arg = j; }
            }
            D_cur[mid] = best; Aat(kp, mid) = static_cast<int>(arg);
            if (mid > mlo) dc(kp, mlo, mid - 1, jlo, arg);
            dc(kp, mid + 1, mhi, arg, jhi);
        };

    auto record = [&](size_t bits, size_t k) {
        std::vector<float> c; size_t m = n;
        for (size_t kk = k;; --kk) {                     // backtrack k clusters
            int j = Aat(kk, m);
            c.push_back(static_cast<float>(cen(static_cast<size_t>(j), m)));
            m = static_cast<size_t>(j);
            if (kk == 1) break;
        }
        std::reverse(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        std::vector<double> cd(c.begin(), c.end());
        nudge_strictly_increasing(cd);
        c.assign(cd.begin(), cd.end());
        const size_t k_target = size_t(1) << bits;       // pad to 2^bits (GPU upload)
        while (c.size() < k_target) c.push_back(c.back());
        R.codebooks[bits].centroids = c;
        R.codebooks[bits].num_entries = c.size();
    };

    size_t kp = 1;
    for (size_t bits = 1; bits <= max_bits; ++bits) {
        const size_t k = size_t(1) << bits;
        if (k >= ndist) {                                // more clusters than distinct values
            std::vector<float> c;
            for (size_t i = 0; i < n; ++i) if (i == 0 || a[i] != a[i - 1]) c.push_back(static_cast<float>(a[i]));
            while (c.size() < k) c.push_back(c.back());
            R.codebooks[bits].centroids = c;
            R.codebooks[bits].num_entries = c.size();
            R.costs[bits] = 0.f;
            continue;
        }
        while (kp < k) {                                 // advance DP to level k (reused across bit-rates)
            ++kp;
            std::fill(D_cur.begin(), D_cur.end(), INF);
            dc(kp, kp, n, kp - 1, n - 1);
            D_prev.swap(D_cur);
        }
        R.costs[bits] = static_cast<float>(D_prev[n] / static_cast<double>(n));
        record(bits, k);
    }
    return R;
}

std::vector<CodebookResult> build_all_dims(const FloatRowMat& data, const LloydOpts& opts) {
    const size_t Nrows = static_cast<size_t>(data.rows());
    const size_t D = static_cast<size_t>(data.cols());
    std::vector<CodebookResult> out(D);
#ifdef SAQ_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long d = 0; d < static_cast<long long>(D); ++d) {
        std::vector<float> col(Nrows);
        for (size_t i = 0; i < Nrows; ++i) {
            col[i] = data(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(d));
        }
        out[static_cast<size_t>(d)] = build_codebook_lloyd(col, opts);
    }
    return out;
}

std::vector<CodebookResult> build_all_dims_exact(const FloatRowMat& data, size_t max_bits) {
    const size_t Nrows = static_cast<size_t>(data.rows());
    const size_t D = static_cast<size_t>(data.cols());
    std::vector<CodebookResult> out(D);
#ifdef SAQ_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (long long d = 0; d < static_cast<long long>(D); ++d) {
        std::vector<float> col(Nrows);
        for (size_t i = 0; i < Nrows; ++i) {
            col[i] = data(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(d));
        }
        out[static_cast<size_t>(d)] = build_codebook_exact(col, max_bits);
    }
    return out;
}

float codebook_mse(std::span<const float> values, const DimensionCodebook& cb) {
    const size_t n = values.size();
    if (n == 0 || cb.num_entries == 0) return 0.f;
    double sse = 0.0;
    for (float v : values) {
        int idx = cb.nearest(v);
        double diff = double(v) - cb.centroid_value(idx);
        sse += diff * diff;
    }
    return static_cast<float>(sse / static_cast<double>(n));
}

}  // namespace saq
