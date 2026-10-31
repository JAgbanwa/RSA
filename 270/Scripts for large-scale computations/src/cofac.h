// cofac.h -- exact norm computation and cofactorisation of sieve survivors (host, multithreaded).
#pragma once
#include <string>
#include <vector>
#include "core.h"
#include "host.h"

struct CofacParams {
    uint64_t lim[2];                    // factor-base bounds (for the size rule)
    int lpb[2];
    int mfb[2];
    int sqside;
    int ecm_effort = 1;                 // multiplies the ECM curve budget
};

struct CofacStats {
    uint64_t tested = 0, bad_ab = 0, rejected[2] = {0, 0}, relations = 0;
};

// Builds CADO-format relations "a,b:p0,..:p1,.." for the survivors of one special-q.
// divs must be sorted by sid. Thread-safe (no shared state besides the inputs).
void cofactor_range(const Poly& P, const QLattice& L, uint64_t q, const CofacParams& cp,
                    const std::vector<Survivor>& surv, const std::vector<Divisor>& divs,
                    size_t s_begin, size_t s_end, std::vector<std::string>& rels, CofacStats& st);
