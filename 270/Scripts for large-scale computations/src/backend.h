// backend.h -- sieving backends (CUDA and CPU reference) behind one interface.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "core.h"
#include "host.h"

struct QSetup {
    Geom g;
    SideQ side[2];
    QLattice L;
    uint64_t q;
    uint32_t r;
};

struct QResult {
    std::vector<Survivor> surv;
    std::vector<Divisor> divs;          // sorted by sid
    uint64_t lost_updates = 0, lost_surv = 0, lost_divs = 0;
    uint64_t updates[2] = {0, 0};
    double max_fill[2] = {0, 0};        // max bucket fill / capacity
};

struct BackendConfig {
    double bkmult = 1.25;               // bucket capacity safety factor
    double mem_gb = 0;                  // bucket memory budget per slice (0 = auto)
    uint32_t surv_cap = 1u << 20;       // survivors per special-q
    uint32_t div_cap = 1u << 24;        // divisor records per special-q
    int device = 0;
};

class Backend {
public:
    virtual ~Backend() {}
    virtual std::string name() const = 0;
    // Upload factor bases. g only fixes I, J and the region size.
    virtual void init(const FBHost fb[2], const Geom& g, const BackendConfig& cfg) = 0;
    virtual void sieve_q(const QSetup& s, QResult& out) = 0;
};

std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_gpu_backend();   // returns nullptr if built without CUDA
bool gpu_backend_available();

// Shared slicing plan so both backends walk the sieve area identically.
struct SlicePlan {
    uint32_t rows_per_slice;            // multiple of rows_per_region
    uint32_t nslices;
    uint64_t regions_per_slice;
    uint32_t cap[2];                    // bucket capacity per region per side
};
SlicePlan plan_slices(const FBHost fb[2], const Geom& g, double bkmult, double mem_bytes);
