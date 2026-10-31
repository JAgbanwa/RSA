// core.h -- sieve logic shared by the CUDA kernels and the CPU reference path.
//
// Every function here is compiled twice: once for the GPU (inside the CUDA
// kernels in backend_gpu.cu) and once for the host (backend_cpu.cpp). The
// phases are written as "for (k = tid; k < n; k += nthreads)" loops, so the
// CPU path simply calls each phase with tid = 0, nthreads = 1, and the atomic
// helpers degrade to plain read-modify-write. Testing the CPU path therefore
// tests the exact arithmetic the GPU runs.
//
// Coordinates: the special-q lattice is spanned by u = (a0,b0), v = (a1,b1);
// a sieve cell (i, j) maps to (a, b) = i*u + j*v with -I/2 <= i < I/2 and
// 0 <= j < J. We store i' = i + I/2 in [0, I). The linear cell index is
// x = j*I + i', split into regions of 2^logreg cells that fit in shared memory.
#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__ inline
#else
#define HD inline
#endif

#define GS_MAXDEG 8
#define GS_MAXSURV 256         // survivors kept per region (overflow is counted)
#define GS_TINY_Q 256          // entries with q < this are sieved cooperatively
#define R_INF 0xFFFFFFFEu      // hits only rows j == 0 mod q (projective in lattice), q >= 2^16
#define R_SKIP 0xFFFFFFFFu     // entry unusable for this special-q
#define R_PROJ 0x80000000u     // q < 2^16 only: R_PROJ | R' means hits where j == R'*i (mod q)

struct SideQ {
    float coef[GS_MAXDEG + 1]; // coefficient of x^m y^(d-m), normalised so max |coef| = 1
    int deg;
    float E;                   // log2 of normalisation, minus log2(q) on the special-q side
    float scale;               // byte units per bit
    float thresh;              // cell survives on this side if lognorm - sieved <= thresh
};

struct Geom {
    uint32_t logI, I, halfI, J;
    uint32_t logreg, reg;
};

struct Update {                // one bucket-sieve hit
    uint32_t p;                // base prime (kept for resieving)
    uint32_t offlog;           // offset in region (low 16 bits) | logp << 16
};

struct Survivor { int32_t i; int32_t j; };
struct Divisor { uint32_t sid; uint32_t p; };   // sid bit 31 = side

// Per-side factor base view (device or host pointers).
struct SideFB {
    const uint32_t* q;         // q = p^k
    const uint32_t* p;         // base prime p
    const uint32_t* R;         // lattice root for the current special-q
    const uint8_t* logp;       // scaled log contribution for the current special-q
    uint32_t n_tiny;           // [0, n_tiny): q < GS_TINY_Q
    uint32_t n_small;          // [n_tiny, n_small): q < reg   (sieved in region)
    uint32_t n_medium;         // [n_small, n_medium): reg <= q < I (line bucket fill)
    uint32_t n;                // [n_medium, n): q >= max(reg, I) (Franke-Kleinjung)
};

// ---------------------------------------------------------------- atomics
#if defined(__CUDA_ARCH__)
__device__ inline uint32_t atom_add(uint32_t* p, uint32_t v) { return atomicAdd(p, v); }
__device__ inline void atom_or(uint32_t* p, uint32_t v) { atomicOr(p, v); }
#else
inline uint32_t atom_add(uint32_t* p, uint32_t v) { uint32_t o = *p; *p = o + v; return o; }
inline void atom_or(uint32_t* p, uint32_t v) { *p |= v; }
#endif

// Add v to byte 'off' of a 4-byte-aligned byte array (little endian).
// Byte totals stay below 256 by construction (see scale choice in main.cpp),
// so a carry never spills into the neighbouring byte.
HD void sadd(uint8_t* S, uint32_t off, uint32_t v) {
    atom_add(reinterpret_cast<uint32_t*>(S + (off & ~3u)), v << (8u * (off & 3u)));
}

// ---------------------------------------------------------------- arithmetic
HD uint64_t mod_s64(int64_t v, uint64_t q) {
    int64_t m = v % (int64_t)q;
    return (uint64_t)(m < 0 ? m + (int64_t)q : m);
}

HD uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

// Inverse of a modulo q (q < 2^32), or 0 if gcd(a, q) != 1.
HD uint64_t inv_mod(uint64_t a, uint64_t q) {
    int64_t t = 0, nt = 1, r = (int64_t)q, nr = (int64_t)(a % q);
    while (nr != 0) {
        int64_t k = r / nr, tmp;
        tmp = t - k * nt; t = nt; nt = tmp;
        tmp = r - k * nr; r = nr; nr = tmp;
    }
    if (r != 1) return 0;
    return (uint64_t)(t < 0 ? t + (int64_t)q : t);
}

// Lattice root of a factor-base entry: hits are i == R*j (mod q).
// CADO convention: root r < q is affine (a == r*b mod q); r >= q is projective
// with r' = r - q (b == r'*a mod q), e.g. r == q means b == 0 mod q.
HD uint32_t lattice_root(uint32_t qq, uint32_t pp, uint32_t r,
                         int64_t a0, int64_t b0, int64_t a1, int64_t b1) {
    uint64_t q = qq, d0, d1;
    if (r < qq) {
        d0 = (mod_s64(a0, q) + q - ((uint64_t)r * mod_s64(b0, q)) % q) % q;
        d1 = (mod_s64(a1, q) + q - ((uint64_t)r * mod_s64(b1, q)) % q) % q;
    } else {
        uint64_t rp = (uint64_t)(r - qq);
        d0 = (mod_s64(b0, q) + q - (rp * mod_s64(a0, q)) % q) % q;
        d1 = (mod_s64(b1, q) + q - (rp * mod_s64(a1, q)) % q) % q;
    }
    uint64_t inv = inv_mod(d0, q);
    if (inv != 0) return (uint32_t)((q - (d1 * inv) % q) % q);
    // d0 not a unit: hits satisfy j*d1 == -i*d0 (mod q); needs d1 to be a unit
    // (p | d0 and p | d1 only happens when p is the special-q itself).
    if (d1 % pp == 0) return R_SKIP;
    if (qq < 65536u) {
        uint64_t i1 = inv_mod(d1, q);
        return R_PROJ | (uint32_t)((q - (d0 * i1) % q) % q);
    }
    return d0 == 0 ? R_INF : R_SKIP;
}

// Franke-Kleinjung reduced basis for the lattice {(i,j) : i == R*j mod q}, q >= I.
// Returns (al,be), (ga,de) with -I < al <= 0 <= ga < I, ga - al >= I, be, de > 0.
HD bool fk_basis(uint32_t q, uint32_t R, int64_t I,
                 int32_t& al, int32_t& be, int32_t& ga, int32_t& de) {
    int64_t px = -(int64_t)q, py = 0, cx = R, cy = 1;
    while ((cx < 0 ? -cx : cx) >= I) {
        int64_t ap = px < 0 ? -px : px, ac = cx < 0 ? -cx : cx;
        int64_t k = ap / ac;
        int64_t nx = px + k * cx, ny = py + k * cy;
        px = cx; py = cy; cx = nx; cy = ny;
        if (cx == 0) return false;
    }
    int64_t ap = px < 0 ? -px : px, ac = cx < 0 ? -cx : cx;
    if (ac == 0) return false;
    int64_t m = (ap - (I - ac)) / ac;
    px += m * cx; py += m * cy;
    if (px <= 0) { al = (int32_t)px; be = (int32_t)py; ga = (int32_t)cx; de = (int32_t)cy; }
    else         { al = (int32_t)cx; be = (int32_t)cy; ga = (int32_t)px; de = (int32_t)py; }
    return be > 0 && de > 0 && al <= 0 && ga >= 0 && (ga - al) >= I;
}

// log2 for a > 0 using only frexpf, IEEE + - * / (no library log), so host and
// device produce bit-identical results when FMA contraction is off (see Makefile).
// Error < 1e-6 bits: log2(m) = (2/ln 2) * atanh(s), s = (m-1)/(m+1), m in [1/sqrt2, sqrt2).
HD float gs_log2f(float a) {
    int e;
    float m = frexpf(a, &e);                // a = m * 2^e, m in [0.5, 1)
    if (m < 0.70710678f) { m *= 2.0f; e -= 1; }
    float s = (m - 1.0f) / (m + 1.0f), s2 = s * s;
    float p = 1.0f / 11.0f;
    p = p * s2 + 1.0f / 9.0f;
    p = p * s2 + 1.0f / 7.0f;
    p = p * s2 + 1.0f / 5.0f;
    p = p * s2 + 1.0f / 3.0f;
    p = p * s2 + 1.0f;
    return (float)e + 2.8853900817779268f * s * p;   // 2/ln 2
}

// Scaled log2 |F_q(i, j)| using the normalised lattice polynomial.
HD float lognorm_units(const SideQ& s, float x, float y) {
    float v = s.coef[s.deg], yp = 1.0f;
    for (int m = s.deg - 1; m >= 0; --m) { yp *= y; v = v * x + s.coef[m] * yp; }
    float a = fabsf(v);
    if (!(a > 1e-30f)) return 0.0f;
    float l = (gs_log2f(a) + s.E) * s.scale;
    return l > 0.0f ? l : 0.0f;
}

// ---------------------------------------------------------------- per-q setup
// Thread e of nthreads: lattice roots and log contributions for this special-q.
HD void phase_roots(uint32_t tid, uint32_t nt, uint32_t n,
                    const uint32_t* q, const uint32_t* p, const uint32_t* r, const float* logbits,
                    uint32_t* R, uint8_t* logp, float scale,
                    int64_t a0, int64_t b0, int64_t a1, int64_t b1) {
    for (uint32_t e = tid; e < n; e += nt) {
        R[e] = lattice_root(q[e], p[e], r[e], a0, b0, a1, b1);
        float l = logbits[e] * scale + 0.5f;
        logp[e] = (uint8_t)(l > 255.0f ? 255.0f : l);
    }
}

// ---------------------------------------------------------------- bucket fill
// Bucket entries (q >= reg) are primes; projective-in-lattice ones are skipped (rare).
HD bool r_special(uint32_t Q, uint32_t Rv) { return Rv >= R_INF || (Q < 65536u && Rv >= R_PROJ); }

struct BucketSet {             // buckets of the current slice, one side
    Update* bk;                // [nregions * cap]
    uint32_t* cnt;             // [nregions]
    uint32_t cap;
    uint64_t region0;          // first region of the slice
};

HD void emit(const Geom& g, const BucketSet& b, uint32_t j, uint32_t ip, uint32_t p, uint32_t lp) {
    uint64_t x = (uint64_t)j * g.I + ip;
    uint64_t rel = (x >> g.logreg) - b.region0;
    uint32_t k = atom_add(&b.cnt[rel], 1u);
    if (k < b.cap) {
        Update u; u.p = p; u.offlog = (uint32_t)(x & (g.reg - 1)) | (lp << 16);
        b.bk[rel * b.cap + k] = u;
    }
}

// Medium entries (reg <= q < I): one work item per (entry, row) of the slice.
HD void fill_medium_item(uint64_t item, const Geom& g, const SideFB& fb, const BucketSet& b,
                         uint32_t j0, uint32_t nrows) {
    uint32_t e = fb.n_small + (uint32_t)(item / nrows);
    uint32_t j = j0 + (uint32_t)(item % nrows);
    if (j == 0) return;
    uint32_t Q = fb.q[e], Rv = fb.R[e];
    if (r_special(Q, Rv)) return;
    uint32_t lp = fb.logp[e], p = fb.p[e];
    uint64_t ip = ((uint64_t)Rv * j + g.halfI) % Q;
    for (; ip < g.I; ip += Q) emit(g, b, j, (uint32_t)ip, p, lp);
}

// Large entries (q >= I): Franke-Kleinjung walk; 'st' keeps the walk position
// (i', j) between slices. Rows [j0, j1) belong to the current slice.
HD void fill_large_one(uint32_t e, const Geom& g, const SideFB& fb, const BucketSet& b,
                       int32_t* st, uint32_t j0, uint32_t j1) {
    uint32_t Q = fb.q[e], Rv = fb.R[e];
    if (r_special(Q, Rv)) return;
    uint32_t lp = fb.logp[e], p = fb.p[e];
    if (Rv == 0) {             // hits exactly the i = 0 column
        for (uint32_t j = (j0 > 0 ? j0 : 1); j < j1; ++j) emit(g, b, j, g.halfI, p, lp);
        return;
    }
    int32_t al, be, ga, de;
    if (!fk_basis(Q, Rv, (int64_t)g.I, al, be, ga, de)) return;
    int32_t ip = st[2 * e], j = st[2 * e + 1];
    const int32_t I = (int32_t)g.I;
    while ((uint32_t)j < j1) {
        if (j > 0 && (uint32_t)j >= j0) emit(g, b, (uint32_t)j, (uint32_t)ip, p, lp);
        if (ip >= -al)          { ip += al; j += be; }
        else if (ip < I - ga)   { ip += ga; j += de; }
        else                    { ip += al + ga; j += be + de; }
    }
    st[2 * e] = ip; st[2 * e + 1] = j;
}

// ---------------------------------------------------------------- region sieve
struct RegionScratch {         // shared memory on the GPU, a local buffer on the CPU
    uint8_t* S[2];             // sieved log sums per side, reg bytes each
    uint32_t* bitmap;          // reg/32 words: survivor flags
    uint32_t* surv;            // GS_MAXSURV cell offsets
    uint32_t* misc;            // [0] = local survivor count, [1] = global base, [2] = n kept
};

struct RegionOut {
    Survivor* surv; uint32_t* nsurv; uint32_t surv_cap;
    Divisor* div; uint32_t* ndiv; uint32_t div_cap;
    uint32_t* lost;            // [0] survivors lost to caps, [1] divisors lost
};

HD void region_rows(const Geom& g, uint64_t region, uint32_t& j0, uint32_t& ip0,
                    uint32_t& nrows, uint32_t& len) {
    uint64_t x0 = region << g.logreg;
    j0 = (uint32_t)(x0 >> g.logI);
    if (g.logI >= g.logreg) { ip0 = (uint32_t)(x0 & (g.I - 1)); nrows = 1; len = g.reg; }
    else { ip0 = 0; nrows = g.reg >> g.logI; len = g.I; }
}

HD void phase_zero(uint32_t tid, uint32_t nt, const Geom& g, RegionScratch& s) {
    uint32_t* a = reinterpret_cast<uint32_t*>(s.S[0]);
    uint32_t* c = reinterpret_cast<uint32_t*>(s.S[1]);
    for (uint32_t k = tid; k < g.reg / 4; k += nt) { a[k] = 0; c[k] = 0; }
    for (uint32_t k = tid; k < g.reg / 32; k += nt) s.bitmap[k] = 0;
    if (tid == 0) { s.misc[0] = 0; s.misc[1] = 0; s.misc[2] = 0; }
}

// First offset in a row segment starting at i' = ip0 that is a hit for (Q, Rv) on row j.
HD uint32_t first_hit(uint32_t Q, uint32_t Rv, uint32_t j, uint32_t halfI, uint32_t ip0) {
    uint32_t t = (uint32_t)(((uint64_t)Rv * (j % Q)) % Q);
    t = (t + halfI % Q) % Q;
    return (t + Q - ip0 % Q) % Q;
}

// Hits of a small entry (q < reg) on row j, in a segment starting at i' = ip0:
// offsets pos, pos + step, ... Returns false if the row has none.
HD bool row_hits(uint32_t Q, uint32_t Rv, uint32_t j, uint32_t halfI, uint32_t ip0,
                 uint32_t& pos, uint32_t& step) {
    if (Rv < R_PROJ) { step = Q; pos = first_hit(Q, Rv, j, halfI, ip0); return true; }
    if (Rv == R_SKIP) return false;
    // projective in the (i,j) plane: j == R'*i (mod Q). With g = gcd(R', Q):
    // rows with g | j, then i == (j/g) * (R'/g)^-1 (mod Q/g).
    uint32_t Rp = (Rv == R_INF) ? 0u : (Rv & ~R_PROJ);
    uint32_t g = gcd_u32(Rp, Q);
    if (j % g) return false;
    uint32_t Qg = Q / g;
    if (Qg == 1) { pos = 0; step = 1; return true; }
    uint32_t inv = (uint32_t)inv_mod((Rp / g) % Qg, Qg);
    uint32_t i0 = (uint32_t)(((uint64_t)((j / g) % Qg) * inv) % Qg);   // i == i0 (mod Qg)
    uint32_t t = (i0 + halfI % Qg) % Qg;                                 // i' = i + I/2
    pos = (t + Qg - ip0 % Qg) % Qg;
    step = Qg;
    return true;
}

HD void phase_small(uint32_t tid, uint32_t nt, const Geom& g, const SideFB& fb,
                    uint8_t* S, uint64_t region) {
    uint32_t j0, ip0, nrows, len;
    region_rows(g, region, j0, ip0, nrows, len);
    // tiny q: every thread takes every nt-th hit of the same entry
    for (uint32_t e = 0; e < fb.n_tiny; ++e) {
        uint32_t Q = fb.q[e], Rv = fb.R[e], lp = fb.logp[e], pos, step;
        for (uint32_t r = 0; r < nrows; ++r) {
            if (!row_hits(Q, Rv, j0 + r, g.halfI, ip0, pos, step)) continue;
            uint32_t base = r * len;
            for (uint32_t k = pos + tid * step; k < len; k += nt * step) sadd(S, base + k, lp);
        }
    }
    // small q: one entry per thread
    for (uint32_t e = fb.n_tiny + tid; e < fb.n_small; e += nt) {
        uint32_t Q = fb.q[e], Rv = fb.R[e], lp = fb.logp[e], pos, step;
        for (uint32_t r = 0; r < nrows; ++r) {
            if (!row_hits(Q, Rv, j0 + r, g.halfI, ip0, pos, step)) continue;
            uint32_t base = r * len;
            for (uint32_t k = pos; k < len; k += step) sadd(S, base + k, lp);
        }
    }
}

HD void phase_buckets(uint32_t tid, uint32_t nt, uint8_t* S, const Update* bk, uint32_t cnt,
                      uint32_t mask) {
    for (uint32_t k = tid; k < cnt; k += nt) {
        uint32_t ol = bk[k].offlog;
        sadd(S, ol & mask, ol >> 16);
    }
}

HD void cell_coords(const Geom& g, uint64_t region, uint32_t c, uint32_t& j, uint32_t& ip) {
    uint64_t x = (region << g.logreg) + c;
    j = (uint32_t)(x >> g.logI);
    ip = (uint32_t)(x & (g.I - 1));
}

HD void phase_threshold(uint32_t tid, uint32_t nt, const Geom& g, const SideQ* sq,
                        RegionScratch& s, uint64_t region) {
    const float invhalf = 1.0f / (float)g.halfI, invJ = 1.0f / (float)g.J;
    for (uint32_t c = tid; c < g.reg; c += nt) {
        uint32_t j, ip;
        cell_coords(g, region, c, j, ip);
        if (j == 0 || j >= g.J) continue;
        int32_t i = (int32_t)ip - (int32_t)g.halfI;
        if (((i | (int32_t)j) & 1) == 0) continue;          // gcd(i, j) even: not coprime
        float x = (float)i * invhalf, y = (float)j * invJ;
        if (lognorm_units(sq[1], x, y) - (float)s.S[1][c] > sq[1].thresh) continue;
        if (lognorm_units(sq[0], x, y) - (float)s.S[0][c] > sq[0].thresh) continue;
        if (gcd_u32((uint32_t)(i < 0 ? -i : i), j) != 1) continue;   // (a,b) would not be coprime
        uint32_t idx = atom_add(&s.misc[0], 1u);
        if (idx < GS_MAXSURV) {
            s.surv[idx] = c;
            atom_or(&s.bitmap[c >> 5], 1u << (c & 31));
        }
    }
}

// Single thread: reserve room in the global survivor list.
HD void phase_alloc(const RegionOut& o, RegionScratch& s) {
    uint32_t n = s.misc[0], lost = 0;
    if (n > GS_MAXSURV) { lost = n - GS_MAXSURV; n = GS_MAXSURV; }
    uint32_t base = 0;
    if (n > 0) {
        base = atom_add(o.nsurv, n);
        if (base >= o.surv_cap) { lost += n; n = 0; }
        else if (base + n > o.surv_cap) { lost += base + n - o.surv_cap; n = o.surv_cap - base; }
    }
    if (lost) atom_add(&o.lost[0], lost);
    s.misc[1] = base; s.misc[2] = n;
}

HD void record_div(const RegionOut& o, uint32_t sid, uint32_t side, uint32_t p) {
    uint32_t d = atom_add(o.ndiv, 1u);
    if (d < o.div_cap) { Divisor v; v.sid = sid | (side << 31); v.p = p; o.div[d] = v; }
    else atom_add(&o.lost[1], 1u);
}

HD void phase_write_surv(uint32_t tid, uint32_t nt, const Geom& g, const RegionOut& o,
                         const RegionScratch& s, uint64_t region) {
    uint32_t n = s.misc[2], base = s.misc[1];
    for (uint32_t k = tid; k < n; k += nt) {
        uint32_t j, ip;
        cell_coords(g, region, s.surv[k], j, ip);
        Survivor v; v.i = (int32_t)ip - (int32_t)g.halfI; v.j = (int32_t)j;
        o.surv[base + k] = v;
    }
}

// Collect which factor-base primes hit each survivor (bucket primes and small primes).
HD void phase_resieve(uint32_t tid, uint32_t nt, const Geom& g, const SideFB& fb, uint32_t side,
                      const Update* bk, uint32_t cnt, const RegionOut& o,
                      const RegionScratch& s, uint64_t region) {
    uint32_t n = s.misc[2], base = s.misc[1];
    if (n == 0) return;
    uint32_t mask = g.reg - 1;
    for (uint32_t k = tid; k < cnt; k += nt) {
        uint32_t off = bk[k].offlog & mask;
        if (!(s.bitmap[off >> 5] & (1u << (off & 31)))) continue;
        for (uint32_t t = 0; t < n; ++t)
            if (s.surv[t] == off) { record_div(o, base + t, side, bk[k].p); break; }
    }
    uint64_t total = (uint64_t)n * fb.n_small;
    for (uint64_t w = tid; w < total; w += nt) {
        uint32_t t = (uint32_t)(w / fb.n_small), e = (uint32_t)(w % fb.n_small);
        uint32_t j, ip, pos, step;
        cell_coords(g, region, s.surv[t], j, ip);
        if (row_hits(fb.q[e], fb.R[e], j, g.halfI, ip, pos, step) && pos == 0)
            record_div(o, base + t, side, fb.p[e]);
    }
}
