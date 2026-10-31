// ecm128.cpp -- see ecm128.h
#include "ecm128.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

typedef unsigned __int128 u128;

namespace {

struct Mont {
    u128 n, ninv, one, r2;     // ninv = -n^-1 mod 2^128, one = R mod n, r2 = R^2 mod n

    static void mul_full(u128 a, u128 b, u128& lo, u128& hi) {
        uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
        u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
        u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
        lo = (mid << 64) | (uint64_t)p00;
        hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    }
    // a*b/R mod n for a, b < n < 2^126
    u128 mul(u128 a, u128 b) const {
        u128 lo, hi, mlo, mhi;
        mul_full(a, b, lo, hi);
        u128 m = lo * ninv;
        mul_full(m, n, mlo, mhi);
        u128 t = hi + mhi + (lo != 0);
        return t >= n ? t - n : t;
    }
    u128 add(u128 a, u128 b) const { u128 s = a + b; return s >= n ? s - n : s; }
    u128 sub(u128 a, u128 b) const { return a >= b ? a - b : a + n - b; }
    u128 to(u128 a) const { return mul(a % n, r2); }
    u128 from(u128 a) const { return mul(a, 1); }

    explicit Mont(const mpz_class& N) {
        n = 0;
        mpz_class t = N;
        n = ((u128)mpz_getlimbn(t.get_mpz_t(), 1) << 64) | (u128)mpz_getlimbn(t.get_mpz_t(), 0);
        if (mpz_size(t.get_mpz_t()) < 2) n = (u128)mpz_getlimbn(t.get_mpz_t(), 0);
        u128 x = n;                                   // n*n == 1 mod 8
        for (int k = 0; k < 7; ++k) x *= 2 - n * x;   // Newton: doubles correct bits
        ninv = (u128)0 - x;
        mpz_class R = 1; R <<= 128;
        mpz_class o = R % N, rr = (R * R) % N;
        auto get = [](const mpz_class& v) {
            u128 r = (u128)mpz_getlimbn(v.get_mpz_t(), 0);
            if (mpz_size(v.get_mpz_t()) > 1) r |= (u128)mpz_getlimbn(v.get_mpz_t(), 1) << 64;
            return r;
        };
        one = get(o); r2 = get(rr);
    }
};

mpz_class to_mpz(u128 v) {
    mpz_class r = (unsigned long)(uint64_t)(v >> 64);
    r <<= 64;
    r += (unsigned long)(uint64_t)v;
    return r;
}

struct Pt { u128 X, Z; };

struct Curve {
    const Mont& M;
    u128 a24;                  // (A+2)/4 in Montgomery form
    Pt dbl(const Pt& P) const {
        u128 s = M.add(P.X, P.Z), d = M.sub(P.X, P.Z);
        u128 s2 = M.mul(s, s), d2 = M.mul(d, d), t = M.sub(s2, d2);
        return {M.mul(s2, d2), M.mul(t, M.add(d2, M.mul(a24, t)))};
    }
    // P + Q given D = P - Q
    Pt add(const Pt& P, const Pt& Q, const Pt& D) const {
        u128 u = M.mul(M.sub(P.X, P.Z), M.add(Q.X, Q.Z));
        u128 v = M.mul(M.add(P.X, P.Z), M.sub(Q.X, Q.Z));
        u128 a = M.add(u, v), b = M.sub(u, v);
        return {M.mul(D.Z, M.mul(a, a)), M.mul(D.X, M.mul(b, b))};
    }
    Pt mul(const Pt& P, uint64_t k) const {
        if (k == 1) return P;
        Pt R0 = P, R1 = dbl(P);
        int top = 63 - __builtin_clzll(k);
        for (int b = top - 1; b >= 0; --b) {
            if ((k >> b) & 1) { R0 = add(R1, R0, P); R1 = dbl(R1); }
            else              { R1 = add(R0, R1, P); R0 = dbl(R0); }
        }
        return R0;
    }
};

// primes up to a bound, shared by all threads
struct PrimeTable {
    std::vector<uint32_t> primes;
    std::vector<uint8_t> isprime;
    uint32_t limit = 0;
    void ensure(uint32_t lim) {
        if (lim <= limit) return;
        isprime.assign(lim + 1, 1);
        isprime[0] = isprime[1] = 0;
        for (uint32_t i = 2; (uint64_t)i * i <= lim; ++i)
            if (isprime[i]) for (uint32_t k = i * i; k <= lim; k += i) isprime[k] = 0;
        primes.clear();
        for (uint32_t i = 2; i <= lim; ++i) if (isprime[i]) primes.push_back(i);
        limit = lim;
    }
};
PrimeTable g_pt;
std::once_flag g_pt_once;

bool try_gcd(const mpz_class& N, u128 v, mpz_class& f) {
    mpz_class g = to_mpz(v);
    mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), N.get_mpz_t());
    if (g != 1 && g != N) { f = g; return true; }
    return false;
}

// one curve; returns 1 = factor found, 0 = nothing, -1 = curve degenerate
int one_curve(const Mont& M, const mpz_class& N, uint64_t sigma, uint32_t B1, uint32_t B2, mpz_class& f) {
    // Suyama: u = s^2 - 5, v = 4s, x0 = u^3, z0 = v^3, a24 = (v-u)^3 (3u+v) / (16 u^3 v)
    u128 s = M.to(sigma), five = M.to(5), four = M.to(4), three = M.to(3), sixteen = M.to(16);
    u128 u = M.sub(M.mul(s, s), five), v = M.mul(four, s);
    u128 u3 = M.mul(M.mul(u, u), u), v3 = M.mul(M.mul(v, v), v);
    u128 vmu = M.sub(v, u), num = M.mul(M.mul(M.mul(vmu, vmu), vmu), M.add(M.mul(three, u), v));
    u128 den = M.mul(M.mul(sixteen, u3), v);
    mpz_class dz = to_mpz(M.from(den)), inv;
    if (!mpz_invert(inv.get_mpz_t(), dz.get_mpz_t(), N.get_mpz_t())) {
        mpz_gcd(dz.get_mpz_t(), dz.get_mpz_t(), N.get_mpz_t());
        if (dz != 1 && dz != N) { f = dz; return 1; }
        return -1;
    }
    u128 invm = 0;
    {
        u128 r = (u128)mpz_getlimbn(inv.get_mpz_t(), 0);
        if (mpz_size(inv.get_mpz_t()) > 1) r |= (u128)mpz_getlimbn(inv.get_mpz_t(), 1) << 64;
        invm = M.to(r);
    }
    Curve C{M, M.mul(num, invm)};
    Pt P{u3, v3};
    // stage 1
    const auto& pr = g_pt.primes;
    for (size_t k = 0; k < pr.size() && pr[k] <= B1; ++k) {
        uint64_t q = pr[k];
        while (q * pr[k] <= B1) q *= pr[k];
        P = C.mul(P, q);
    }
    if (try_gcd(N, P.Z, f)) return 1;
    // stage 2: primes p = k*D +/- j in (B1, B2]
    const uint32_t D = 210;
    std::vector<Pt> baby;            // j*P for odd j < D/2 with gcd(j, D) = 1
    std::vector<uint32_t> js;
    Pt P2 = C.dbl(P), prev = P, cur = C.add(P2, P, P);   // P, 3P
    std::vector<Pt> odd{P, cur};                          // 1P, 3P, ...
    for (uint32_t j = 5; j < D / 2; j += 2) {
        Pt nx = C.add(cur, P2, prev);
        prev = cur; cur = nx; odd.push_back(cur);
    }
    for (uint32_t j = 1, idx = 0; j < D / 2; j += 2, ++idx)
        if (j % 3 && j % 5 && j % 7) { baby.push_back(odd[idx]); js.push_back(j); }
    Pt G1 = C.mul(P, D), G = G1, Gprev = G1;             // G = k*D*P, Gprev = (k-1)*D*P
    uint32_t k = 1;
    auto advance = [&]() {
        Pt nx = (k == 1) ? C.dbl(G1) : C.add(G, G1, Gprev);
        Gprev = G; G = nx; ++k;
    };
    while ((uint64_t)(k + 1) * D < B1) advance();          // giant steps up to just below B1
    u128 acc = M.one;
    const auto& isp = g_pt.isprime;
    for (; (uint64_t)k * D <= (uint64_t)B2 + D; advance()) {
        for (size_t t = 0; t < baby.size(); ++t) {
            uint64_t lo = (uint64_t)k * D - js[t], hi = (uint64_t)k * D + js[t];
            bool use = (lo > B1 && lo <= B2 && isp[lo]) || (hi > B1 && hi <= B2 && isp[hi]);
            if (!use) continue;
            u128 d = M.sub(M.mul(G.X, baby[t].Z), M.mul(baby[t].X, G.Z));
            acc = M.mul(acc, d);
        }
    }
    if (try_gcd(N, acc, f)) return 1;
    return 0;
}

}  // namespace

bool ecm128_factor(const mpz_class& n, double B1base, int ncurves, mpz_class& f) {
    if (mpz_sizeinbase(n.get_mpz_t(), 2) > 125 || mpz_even_p(n.get_mpz_t())) return false;
    std::call_once(g_pt_once, [] { g_pt.ensure(4000000); });
    Mont M(n);
    for (int t = 0; t < ncurves; ++t) {
        uint32_t B1 = (uint32_t)(B1base * (1.0 + 0.25 * t));
        uint32_t B2 = std::min<uint32_t>(50u * B1, g_pt.limit - 1000);
        int r = one_curve(M, n, 7 + (uint64_t)t * 11, B1, B2, f);
        if (r == 1) return true;
    }
    return false;
}
