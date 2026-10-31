// cofac.cpp -- turn sieve survivors into verified relations.
#include "cofac.h"
#include "ecm128.h"
#include <ecm.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

typedef unsigned __int128 u128;

namespace {
inline uint64_t mulm(uint64_t a, uint64_t b, uint64_t n) { return (uint64_t)((u128)a * b % n); }

uint64_t gcd64(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }

// Pollard-Brent rho for n < 2^63 composite, returns a non-trivial factor or 0.
uint64_t rho64(uint64_t n, uint64_t seed) {
    if (n % 2 == 0) return 2;
    for (uint64_t c = seed; c < seed + 20; ++c) {
        uint64_t y = 2, x, q = 1, g = 1, ys = 0;
        uint64_t m = 128, r = 1;
        do {
            x = y;
            for (uint64_t i = 0; i < r; ++i) y = (mulm(y, y, n) + c) % n;
            uint64_t k = 0;
            do {
                ys = y;
                for (uint64_t i = 0; i < std::min(m, r - k); ++i) {
                    y = (mulm(y, y, n) + c) % n;
                    q = mulm(q, x > y ? x - y : y - x, n);
                }
                g = gcd64(q, n);
                k += m;
            } while (k < r && g == 1);
            r <<= 1;
        } while (g == 1 && r < (1ull << 26));
        if (g == n) {
            do { ys = (mulm(ys, ys, n) + c) % n; g = gcd64(x > ys ? x - ys : ys - x, n); } while (g == 1);
        }
        if (g != n && g != 1) return g;
    }
    return 0;
}

// Find a factor of composite n whose prime factors are expected to have at most lpb bits.
// Effort is sized to lpb, like CADO's ncurves: B1 ramps from a base suited to lpb-bit factors.
bool find_factor(const mpz_class& n, mpz_class& f, int lpb, int effort) {
    if (mpz_sizeinbase(n.get_mpz_t(), 2) <= 62) {
        uint64_t r = rho64(n.get_ui(), 1);
        if (r) { f = (unsigned long)r; return true; }
    }
    double B1base = lpb <= 22 ? 50 : lpb <= 28 ? 105 : lpb <= 33 ? 200 : lpb <= 37 ? 320 : 500;
    int ncurves = (lpb <= 28 ? 12 : lpb <= 33 ? 18 : 24) * effort;
    if (mpz_sizeinbase(n.get_mpz_t(), 2) <= 125 && mpz_odd_p(n.get_mpz_t()))
        return ecm128_factor(n, B1base, ncurves, f);       // fast path for 2-limb cofactors
    ecm_params ep;
    ecm_init(ep);
    bool ok = false;
    for (int t = 0; t < ncurves && !ok; ++t) {
        mpz_set_ui(ep->sigma, 0);         // random curve each time
        mpz_set_ui(ep->x, 0);
        ep->B1done = 1.0;
        ep->verbose = 0;
        mpz_class g, nn = n;
        double B1 = B1base * (1.0 + 0.25 * t);
        int res = ecm_factor(g.get_mpz_t(), nn.get_mpz_t(), B1, ep);
        if (res > 0 && g != 1 && g != n) { f = g; ok = true; }
    }
    ecm_clear(ep);
    return ok;
}

// Fully factor n into primes of at most lpb bits.
bool split_cofactor(mpz_class n, int lpb, std::vector<mpz_class>& out, int effort) {
    std::vector<mpz_class> todo{n};
    while (!todo.empty()) {
        mpz_class x = todo.back(); todo.pop_back();
        if (x == 1) continue;
        if (mpz_probab_prime_p(x.get_mpz_t(), 25)) {
            if ((int)mpz_sizeinbase(x.get_mpz_t(), 2) > lpb) return false;
            out.push_back(x);
            continue;
        }
        mpz_class f;
        if (!find_factor(x, f, lpb, effort)) return false;
        todo.push_back(f);
        todo.push_back(x / f);
    }
    return true;
}

const unsigned TD_PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};

// Factor |F(a,b)| on one side; listed primes come from the GPU resieve.
bool factor_side(const std::vector<mpz_class>& f, const mpz_class& a, const mpz_class& b,
                 const std::vector<uint32_t>& listed, uint64_t sq, uint64_t lim, int lpb, int mfb, int effort,
                 std::vector<mpz_class>& primes) {
    mpz_class N = eval_homog(f, a, b);
    N = abs(N);
    if (N == 0) return false;
    if (sq) {
        mpz_class Q = (unsigned long)sq;
        if (!mpz_divisible_p(N.get_mpz_t(), Q.get_mpz_t())) return false;   // lattice bug guard
        while (mpz_divisible_p(N.get_mpz_t(), Q.get_mpz_t())) { N /= Q; primes.push_back(Q); }
    }
    for (uint32_t p : listed) {
        if (p == sq) continue;
        while (mpz_divisible_ui_p(N.get_mpz_t(), p)) { mpz_divexact_ui(N.get_mpz_t(), N.get_mpz_t(), p); primes.push_back(mpz_class(p)); }
    }
    for (unsigned p : TD_PRIMES)
        while (mpz_divisible_ui_p(N.get_mpz_t(), p)) { mpz_divexact_ui(N.get_mpz_t(), N.get_mpz_t(), p); primes.push_back(mpz_class(p)); }
    if (N == 1) return true;
    int bits = (int)mpz_sizeinbase(N.get_mpz_t(), 2);
    if (bits > mfb) return false;
    // Every prime left is above the factor-base bound B = lim and at most L = 2^lpb, so a
    // cofactor made of k of them lies in (B^k, L^k]. Reject sizes no k can explain (as CADO does).
    if (lim > 1) {
        double lb = std::log2((double)lim), lc = mpz_sizeinbase(N.get_mpz_t(), 2) - 1
                    + std::log2(mpz_get_d(N.get_mpz_t()) / std::ldexp(1.0, bits - 1));
        bool possible = false;
        for (int k = 1; k * lb < lc + 1e-9 || k == 1; ++k)
            if (lc > k * lb - 1e-9 && lc <= (double)k * lpb + 1e-9) { possible = true; break; }
        if (!possible) return false;
    }
    if (mpz_probab_prime_p(N.get_mpz_t(), 25)) {
        if (bits > lpb) return false;
        primes.push_back(N);
        return true;
    }
    return split_cofactor(N, lpb, primes, effort);
}

void append_hex_list(std::string& s, std::vector<mpz_class>& v) {
    std::sort(v.begin(), v.end());
    char buf[64];
    for (size_t k = 0; k < v.size(); ++k) {
        if (k) s += ',';
        if (mpz_fits_ulong_p(v[k].get_mpz_t())) { snprintf(buf, sizeof buf, "%lx", v[k].get_ui()); s += buf; }
        else s += v[k].get_str(16);
    }
}
}  // namespace

void cofactor_range(const Poly& P, const QLattice& L, uint64_t q, const CofacParams& cp,
                    const std::vector<Survivor>& surv, const std::vector<Divisor>& divs,
                    size_t s_begin, size_t s_end, std::vector<std::string>& rels, CofacStats& st) {
    // divisor index range for each survivor (divs sorted by sid low bits)
    auto lo = std::lower_bound(divs.begin(), divs.end(), (uint32_t)s_begin,
                               [](const Divisor& d, uint32_t v) { return (d.sid & 0x7FFFFFFFu) < v; });
    std::vector<uint32_t> listed[2];
    for (size_t sid = s_begin; sid < s_end; ++sid) {
        listed[0].clear(); listed[1].clear();
        while (lo != divs.end() && (lo->sid & 0x7FFFFFFFu) == sid) {
            listed[lo->sid >> 31].push_back(lo->p);
            ++lo;
        }
        for (auto& v : listed) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); }
        st.tested++;
        const Survivor& s = surv[sid];
        __int128 A = (__int128)s.i * L.a0 + (__int128)s.j * L.a1;
        __int128 B = (__int128)s.i * L.b0 + (__int128)s.j * L.b1;
        if (B < 0) { A = -A; B = -B; }
        if (B == 0 || A > INT64_MAX || A < -INT64_MAX || B > INT64_MAX) { st.bad_ab++; continue; }
        int64_t a = (int64_t)A, b = (int64_t)B;
        uint64_t ua = a < 0 ? (uint64_t)(-a) : (uint64_t)a;
        if (gcd64(ua, (uint64_t)b) != 1) { st.bad_ab++; continue; }
        mpz_class az = (long)a, bz = (long)b;
        std::vector<mpz_class> pr[2];
        bool ok = true;
        for (int side : {cp.sqside, 1 - cp.sqside}) {
            if (!factor_side(P.f[side], az, bz, listed[side], side == cp.sqside ? q : 0,
                             cp.lim[side], cp.lpb[side], cp.mfb[side], cp.ecm_effort, pr[side])) {
                st.rejected[side]++; ok = false; break;
            }
        }
        if (!ok) continue;
        std::string line = std::to_string(a) + "," + std::to_string(b) + ":";
        append_hex_list(line, pr[0]);
        line += ":";
        append_hex_list(line, pr[1]);
        rels.push_back(line);
        st.relations++;
    }
}
