// host.cpp -- polynomials, roots mod p, factor bases, special-q lattice reduction.
#include "host.h"
#include <zlib.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

typedef unsigned __int128 u128;

// ------------------------------------------------------------------ poly file
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

Poly read_poly(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open polynomial file " + path);
    Poly P;
    std::vector<mpz_class> c(GS_MAXDEG + 1, 0), Y(2, 0);
    int maxc = -1, maxy = -1;
    std::vector<mpz_class> polyline[2];
    bool havepl[2] = {false, false};
    std::string line;
    while (std::getline(in, line)) {
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
        if (val.empty()) continue;
        if (key == "n") P.n = mpz_class(val);
        else if (key == "skew") P.skew = atof(val.c_str());
        else if (key.size() == 2 && key[0] == 'c' && isdigit(key[1])) {
            int k = key[1] - '0'; c[k] = mpz_class(val); maxc = std::max(maxc, k);
        } else if (key.size() == 2 && key[0] == 'Y' && isdigit(key[1])) {
            int k = key[1] - '0'; if (k > 1) throw std::runtime_error("only linear Y polynomial supported");
            Y[k] = mpz_class(val); maxy = std::max(maxy, k);
        } else if (key == "poly0" || key == "poly1") {
            int s = key[4] - '0';
            std::stringstream ss(val); std::string tok;
            while (std::getline(ss, tok, ',')) polyline[s].push_back(mpz_class(trim(tok)));
            havepl[s] = true;
        }
    }
    if (havepl[0] && havepl[1]) { P.f[0] = polyline[0]; P.f[1] = polyline[1]; }
    else {
        if (maxc < 1 || maxy < 1) throw std::runtime_error("polynomial file needs c0..cd and Y0,Y1");
        P.f[0] = {Y[0], Y[1]};
        P.f[1].assign(c.begin(), c.begin() + maxc + 1);
    }
    for (int s = 0; s < 2; ++s) {
        while (P.f[s].size() > 1 && P.f[s].back() == 0) P.f[s].pop_back();
        if (P.deg(s) > GS_MAXDEG) throw std::runtime_error("degree too large");
    }
    if (P.n == 0) throw std::runtime_error("polynomial file has no n:");
    // sanity: the two polynomials must share a root mod n (resultant == 0 mod n) -- cheap check via
    // the rational root m = -Y0/Y1 mod n
    {
        mpz_class inv, m, v = 0;
        if (mpz_invert(inv.get_mpz_t(), P.f[0][1].get_mpz_t(), P.n.get_mpz_t())) {
            m = -P.f[0][0] * inv; m %= P.n;
            for (int k = P.deg(1); k >= 0; --k) { v = v * m + P.f[1][k]; v %= P.n; }
            if (v != 0) throw std::runtime_error("polynomials do not share a root mod n");
        }
    }
    return P;
}

mpz_class eval_homog(const std::vector<mpz_class>& f, const mpz_class& a, const mpz_class& b) {
    // F(a,b) = sum f_k a^k b^(d-k), Horner in a with powers of b
    int d = (int)f.size() - 1;
    mpz_class v = f[d], bp = 1;
    for (int k = d - 1; k >= 0; --k) { bp *= b; v = v * a + f[k] * bp; }
    return v;
}

// ------------------------------------------------------------------ roots mod p
namespace {
typedef std::vector<uint64_t> PV;   // coefficients mod p, low degree first

inline uint64_t mulm(uint64_t a, uint64_t b, uint64_t p) { return (uint64_t)((u128)a * b % p); }
inline uint64_t addm(uint64_t a, uint64_t b, uint64_t p) { uint64_t s = a + b; return s >= p ? s - p : s; }
inline uint64_t subm(uint64_t a, uint64_t b, uint64_t p) { return a >= b ? a - b : a + p - b; }
uint64_t powm(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1 % p; a %= p;
    while (e) { if (e & 1) r = mulm(r, a, p); a = mulm(a, a, p); e >>= 1; }
    return r;
}
inline uint64_t invm(uint64_t a, uint64_t p) { return powm(a, p - 2, p); }
void ptrim(PV& a) { while (!a.empty() && a.back() == 0) a.pop_back(); }
int pdeg(const PV& a) { return (int)a.size() - 1; }

void pmonic(PV& a, uint64_t p) {
    uint64_t c = invm(a.back(), p);
    for (auto& x : a) x = mulm(x, c, p);
}
// a mod f, f monic
void pmod(PV& a, const PV& f, uint64_t p) {
    int df = pdeg(f);
    ptrim(a);
    while (pdeg(a) >= df) {
        uint64_t lead = a.back(); int sh = pdeg(a) - df;
        for (int k = 0; k <= df; ++k) a[sh + k] = subm(a[sh + k], mulm(lead, f[k], p), p);
        ptrim(a);
    }
}
PV pmulmod(const PV& a, const PV& b, const PV& f, uint64_t p) {
    if (a.empty() || b.empty()) return PV();
    PV c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i]) for (size_t j = 0; j < b.size(); ++j) c[i + j] = addm(c[i + j], mulm(a[i], b[j], p), p);
    pmod(c, f, p);
    return c;
}
PV ppowmod(PV base, uint64_t e, const PV& f, uint64_t p) {
    PV r{1};
    pmod(base, f, p);
    while (e) { if (e & 1) r = pmulmod(r, base, f, p); base = pmulmod(base, base, f, p); e >>= 1; }
    return r;
}
PV pgcd(PV a, PV b, uint64_t p) {
    ptrim(a); ptrim(b);
    while (!b.empty()) { pmonic(b, p); pmod(a, b, p); std::swap(a, b); }
    if (!a.empty()) pmonic(a, p);
    return a;
}
// exact quotient a / b, b monic
PV pdiv(PV a, const PV& b, uint64_t p) {
    int db = pdeg(b), da = pdeg(a);
    PV q(da - db + 1, 0);
    for (int k = da; k >= db; --k) {
        uint64_t c = a[k]; q[k - db] = c;
        if (c) for (int t = 0; t <= db; ++t) a[k - db + t] = subm(a[k - db + t], mulm(c, b[t], p), p);
    }
    return q;
}
uint64_t rng_state = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }

void split_linear(const PV& g, uint64_t p, std::vector<uint64_t>& out) {
    int d = pdeg(g);
    if (d <= 0) return;
    if (d == 1) { out.push_back(subm(0, g[0], p)); return; }   // g monic: x + g0
    for (int tries = 0; tries < 200; ++tries) {
        PV h{rnd() % p, 1};
        PV t = ppowmod(h, (p - 1) / 2, g, p);
        if (t.empty()) t.push_back(0);
        t[0] = subm(t[0], 1, p);
        PV dd = pgcd(g, t, p);
        if (pdeg(dd) > 0 && pdeg(dd) < d) {
            split_linear(dd, p, out);
            split_linear(pdiv(g, dd, p), p, out);
            return;
        }
    }
}
}  // namespace

std::vector<uint64_t> roots_mod_p(const std::vector<mpz_class>& f, uint64_t p) {
    std::vector<uint64_t> out;
    PV a(f.size());
    mpz_class t;
    for (size_t k = 0; k < f.size(); ++k) {
        t = f[k] % (unsigned long)p; if (t < 0) t += (unsigned long)p;
        a[k] = t.get_ui();
    }
    ptrim(a);
    if (pdeg(a) <= 0) return out;
    if (pdeg(a) == 1) { out.push_back(mulm(subm(0, a[0], p), invm(a[1], p), p)); return out; }
    if (p < 600) {
        for (uint64_t x = 0; x < p; ++x) {
            uint64_t v = 0;
            for (int k = pdeg(a); k >= 0; --k) v = addm(mulm(v, x, p), a[k], p);
            if (v == 0) out.push_back(x);
        }
        return out;
    }
    pmonic(a, p);
    PV xp = ppowmod(PV{0, 1}, p, a, p);
    if (xp.size() < 2) xp.resize(2, 0);
    xp[1] = subm(xp[1], 1, p);
    PV g = pgcd(a, xp, p);
    split_linear(g, p, out);
    std::sort(out.begin(), out.end());
    return out;
}

void primes_upto(uint64_t lim, std::vector<uint32_t>& out) {
    out.clear();
    if (lim < 2) return;
    uint64_t s = (uint64_t)std::sqrt((double)lim) + 2;
    std::vector<char> small(s + 1, 1);
    std::vector<uint32_t> base;
    for (uint64_t i = 2; i <= s; ++i)
        if (small[i]) { base.push_back((uint32_t)i); for (uint64_t k = i * i; k <= s; k += i) small[k] = 0; }
    const uint64_t SEG = 1 << 20;
    std::vector<char> seg(SEG);
    for (uint64_t lo = 2; lo <= lim; lo += SEG) {
        uint64_t hi = std::min(lim + 1, lo + SEG);
        std::fill(seg.begin(), seg.end(), 1);
        for (uint32_t b : base) {
            uint64_t bb = (uint64_t)b * b;
            if (bb >= hi) break;
            uint64_t st = std::max(bb, (lo + b - 1) / b * b);
            for (uint64_t k = st; k < hi; k += b) seg[k - lo] = 0;
        }
        for (uint64_t k = lo; k < hi; ++k) if (seg[k - lo]) out.push_back((uint32_t)k);
    }
}

// ------------------------------------------------------------------ factor bases
void FBHost::sort_and_split(uint32_t reg, uint32_t I) {
    std::vector<size_t> idx(q.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return q[a] < q[b]; });
    auto perm = [&](auto& v) { auto w = v; for (size_t k = 0; k < idx.size(); ++k) v[k] = w[idx[k]]; };
    perm(q); perm(p); perm(r); perm(logbits);
    uint32_t big = std::max(reg, I);
    n_tiny = n_small = n_medium = 0;
    for (size_t k = 0; k < q.size(); ++k) {
        if (q[k] < GS_TINY_Q) n_tiny = (uint32_t)k + 1;
        if (q[k] < reg) n_small = (uint32_t)k + 1;
        if (q[k] < big) n_medium = (uint32_t)k + 1;
    }
    n_small = std::max(n_small, n_tiny);
    n_medium = std::max(n_medium, n_small);
}

static void push_entry(FBHost& fb, uint32_t q, uint32_t p, uint32_t r, float lb) {
    fb.q.push_back(q); fb.p.push_back(p); fb.r.push_back(r); fb.logbits.push_back(lb);
}

static uint64_t eval_mod(const std::vector<mpz_class>& f, uint64_t x, uint64_t m) {
    mpz_class v = 0;
    for (int k = (int)f.size() - 1; k >= 0; --k) { v = v * (unsigned long)x + f[k]; v %= (unsigned long)m; }
    if (v < 0) v += (unsigned long)m;
    return v.get_ui();
}

void build_fb_internal(const std::vector<mpz_class>& f, uint32_t lim, uint32_t powlim, FBHost& fb) {
    std::vector<uint32_t> pr;
    primes_upto(lim, pr);
    int d = (int)f.size() - 1;
    std::vector<mpz_class> df(d);
    for (int k = 1; k <= d; ++k) df[k - 1] = f[k] * k;
    for (uint32_t p : pr) {
        float lb = (float)std::log2((double)p);
        std::vector<uint64_t> rts = roots_mod_p(f, p);
        for (uint64_t r : rts) {
            push_entry(fb, p, p, (uint32_t)r, lb);
            // Hensel-lift unramified roots to prime powers below powlim
            if ((uint64_t)p * p >= powlim) continue;
            if (eval_mod(df, r, p) == 0) continue;
            uint64_t pk = p, rk = r;
            while (pk * p < powlim) {
                uint64_t pk1 = pk * p;
                uint64_t fv = eval_mod(f, rk, pk1), dv = eval_mod(df, rk, pk1);
                // rk <- rk - f(rk)/f'(rk) mod pk1 ; f'(rk) is a unit mod p
                mpz_class inv, m = (unsigned long)pk1, dvz = (unsigned long)dv;
                mpz_invert(inv.get_mpz_t(), dvz.get_mpz_t(), m.get_mpz_t());
                mpz_class nr = mpz_class((unsigned long)rk) - mpz_class((unsigned long)fv) * inv;
                nr %= m; if (nr < 0) nr += m;
                rk = nr.get_ui(); pk = pk1;
                push_entry(fb, (uint32_t)pk, p, (uint32_t)rk, lb);
            }
        }
        if (f[d] % (unsigned long)p == 0) push_entry(fb, p, p, p, lb);   // projective root
    }
}

static bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d = 2; (uint64_t)d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}

void load_fb_cado(const std::string& path, uint32_t lim, uint32_t powlim, FBHost& fb) {
    gzFile in = gzopen(path.c_str(), "rb");
    if (!in) throw std::runtime_error("cannot open factor base " + path);
    std::vector<char> buf(1 << 16);
    while (gzgets(in, buf.data(), (int)buf.size())) {
        const char* s = buf.data();
        if (*s == '#' || *s == '\n' || *s == 0) continue;
        char* e;
        unsigned long q = strtoul(s, &e, 10);
        if (*e != ':') continue;
        s = e + 1;
        int n1 = 1, n0 = 0;
        // either "q: r1,r2" or "q:n1,n0: r1,r2"
        const char* c2 = strchr(s, ':');
        if (c2) { n1 = (int)strtol(s, &e, 10); n0 = (int)strtol(e + 1, &e, 10); s = c2 + 1; }
        if (q > lim && n0 == 0) continue;
        if (n0 > 0 && q >= powlim) continue;
        uint32_t p = (uint32_t)q;
        if (n0 > 0) {                      // prime power: find its base prime
            for (uint32_t t = 2; (uint64_t)t * t <= q; ++t) if (q % t == 0) { p = t; break; }
        }
        float lb = (float)((n1 - n0) * std::log2((double)p));
        while (*s) {
            while (*s == ' ' || *s == ',') ++s;
            if (!isdigit((unsigned char)*s)) break;
            unsigned long r = strtoul(s, &e, 10);
            s = e;
            // r >= q is a projective root r - q (CADO format); kept as-is
            if (r < 2 * q && r < 0xFFFFFFFFul) push_entry(fb, (uint32_t)q, p, (uint32_t)r, lb);
        }
    }
    gzclose(in);
    (void)is_prime_u32;
}

// ------------------------------------------------------------------ special-q lattice
bool reduce_qlattice(uint64_t q, uint64_t r, double skew, QLattice& L) {
    // Lagrange-Gauss reduction of (q,0),(r,1) for the norm a^2 + (s*b)^2
    long double S2 = (long double)skew * skew;
    int64_t ua = (int64_t)q, ub = 0, va = (int64_t)r, vb = 1;
    auto N = [&](int64_t a, int64_t b) { return (long double)a * a + S2 * (long double)b * b; };
    auto D = [&](int64_t a, int64_t b, int64_t c, int64_t e) { return (long double)a * c + S2 * (long double)b * e; };
    if (N(ua, ub) > N(va, vb)) { std::swap(ua, va); std::swap(ub, vb); }
    for (int it = 0; it < 200; ++it) {
        long double mu = std::round(D(ua, ub, va, vb) / N(ua, ub));
        int64_t m = (int64_t)mu;
        va -= m * ua; vb -= m * ub;
        if (N(va, vb) >= N(ua, ub)) break;
        std::swap(ua, va); std::swap(ub, vb);
    }
    L.a0 = ua; L.b0 = ub; L.a1 = va; L.b1 = vb;
    __int128 det = (__int128)L.a0 * L.b1 - (__int128)L.a1 * L.b0;
    if (det < 0) { L.a1 = -L.a1; L.b1 = -L.b1; det = -det; }   // keep det = +q
    return det == (__int128)q;
}

void make_sideq(const std::vector<mpz_class>& f, const QLattice& L, const Geom& g,
                bool is_sqside, uint64_t q, double thresh_bits, double target_units, SideQ& s) {
    int d = (int)f.size() - 1;
    // P(i) = sum_k f_k (a0 i + a1)^k (b0 i + b1)^(d-k); coefficient of i^m is that of i^m j^(d-m)
    std::vector<mpz_class> P(d + 1, 0);
    mpz_class a0 = (long)L.a0, a1 = (long)L.a1, b0 = (long)L.b0, b1 = (long)L.b1;
    for (int k = 0; k <= d; ++k) {
        std::vector<mpz_class> t{f[k]};
        auto mul_lin = [&](const mpz_class& c1, const mpz_class& c0) {   // t *= (c1 i + c0)
            std::vector<mpz_class> u(t.size() + 1, 0);
            for (size_t m = 0; m < t.size(); ++m) { u[m] += t[m] * c0; u[m + 1] += t[m] * c1; }
            t = u;
        };
        for (int e = 0; e < k; ++e) mul_lin(a0, a1);
        for (int e = 0; e < d - k; ++e) mul_lin(b0, b1);
        for (size_t m = 0; m < t.size(); ++m) P[m] += t[m];
    }
    std::vector<double> Em(d + 1);
    double M = 0;
    for (int m = 0; m <= d; ++m) {
        Em[m] = P[m].get_d() * std::pow((double)g.halfI, m) * std::pow((double)g.J, d - m);
        M = std::max(M, std::fabs(Em[m]));
    }
    if (M == 0) M = 1;
    s.deg = d;
    for (int m = 0; m <= GS_MAXDEG; ++m) s.coef[m] = m <= d ? (float)(Em[m] / M) : 0.0f;
    double off = is_sqside ? std::log2((double)q) : 0.0;
    s.E = (float)(std::log2(M) - off);
    double maxlog = std::log2(M) + std::log2((double)d + 1) - off;
    double scale = target_units / std::max(maxlog, 8.0);
    scale = std::min(scale, 240.0 / 32.0);      // keeps every single log p below 255
    s.scale = (float)scale;
    s.thresh = (float)(std::min(thresh_bits, maxlog) * scale);
}
