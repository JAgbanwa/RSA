// backend_common.cpp -- slicing plan shared by the backends.
#include "backend.h"
#include <algorithm>
#include <cmath>

SlicePlan plan_slices(const FBHost fb[2], const Geom& g, double bkmult, double mem_bytes) {
    SlicePlan sp;
    for (int s = 0; s < 2; ++s) {
        // expected bucket updates per region = reg * sum over bucket entries of 1/q
        double inv = 0;
        for (size_t e = fb[s].n_small; e < fb[s].size(); ++e) inv += 1.0 / fb[s].q[e];
        double E = g.reg * inv;
        double cap = bkmult * (E + 8.0 * std::sqrt(E) + 64.0);
        sp.cap[s] = (uint32_t)std::min(cap, 4.0e9);
    }
    uint32_t rows_per_region = g.logI >= g.logreg ? 1 : (g.reg >> g.logI);
    double regions_per_row = g.logI >= g.logreg ? (double)(g.I >> g.logreg) : 1.0 / rows_per_region;
    double bytes_per_region = (double)(sp.cap[0] + sp.cap[1]) * sizeof(Update) + 8.0;
    double max_regions = std::max(1.0, mem_bytes / bytes_per_region);
    double rows = max_regions / regions_per_row;
    uint32_t r = (uint32_t)std::min<double>(rows, g.J);
    r = std::max(r, rows_per_region);
    r -= r % rows_per_region;
    sp.rows_per_slice = r;
    sp.nslices = (g.J + r - 1) / r;
    sp.regions_per_slice = (uint64_t)((double)r * regions_per_row + 0.5);
    if (sp.regions_per_slice == 0) sp.regions_per_slice = 1;
    return sp;
}
