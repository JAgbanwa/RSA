// host.h -- host-only pieces: polynomials, roots mod p, factor bases, special-q lattice.
#pragma once
#include <gmpxx.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "core.h"

struct Poly {
    mpz_class n;
    double skew = 1.0;
    std::vector<mpz_class> f[2];       // f[side][k] = coefficient of x^k; side 0 rational (Y), side 1 algebraic (c)
    int deg(int side) const { return (int)f[side].size() - 1; }
};

Poly read_poly(const std::string& path);
mpz_class eval_homog(const std::vector<mpz_class>& f, const mpz_class& a, const mpz_class& b);

// distinct affine roots of f mod p (p prime, p < 2^63)
std::vector<uint64_t> roots_mod_p(const std::vector<mpz_class>& f, uint64_t p);

void primes_upto(uint64_t lim, std::vector<uint32_t>& out);

struct FBHost {
    std::vector<uint32_t> q, p, r;     // r >= q means projective root
    std::vector<float> logbits;
    uint32_t n_tiny = 0, n_small = 0, n_medium = 0;
    void sort_and_split(uint32_t reg, uint32_t I);
    size_t size() const { return q.size(); }
};

void build_fb_internal(const std::vector<mpz_class>& f, uint32_t lim, uint32_t powlim, FBHost& fb);
void load_fb_cado(const std::string& path, uint32_t lim, uint32_t powlim, FBHost& fb);

struct QLattice { int64_t a0, b0, a1, b1; };
bool reduce_qlattice(uint64_t q, uint64_t r, double skew, QLattice& L);

// SideQ for one side: normalised lattice polynomial, log offset, scale, threshold.
// thresh_bits: a cell survives on this side if the unsieved part of its norm has at most
// this many bits (CADO's lambda*lpb, by default mfb + 0.3*lpb).
void make_sideq(const std::vector<mpz_class>& f, const QLattice& L, const Geom& g,
                bool is_sqside, uint64_t q, double thresh_bits, double target_units, SideQ& s);
