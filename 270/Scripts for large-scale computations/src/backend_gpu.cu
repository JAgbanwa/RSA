// backend_gpu.cu -- CUDA backend. The kernels are thin wrappers around the
// phase functions in core.h, so the GPU runs the same arithmetic as the CPU path.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "backend.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    throw std::runtime_error("CUDA failure"); } } while (0)

namespace {

constexpr int REGION_THREADS = 256;
constexpr int FILL_THREADS = 256;

struct RegionArgs {
    Geom g;
    SideQ sq[2];
    SideFB fb[2];
    const Update* bk[2];
    const uint32_t* cnt[2];
    uint32_t cap[2];
    uint64_t region0;
    RegionOut ro;
};

__global__ void k_roots(uint32_t n, const uint32_t* q, const uint32_t* p, const uint32_t* r,
                        const float* logbits, uint32_t* R, uint8_t* logp, float scale,
                        int64_t a0, int64_t b0, int64_t a1, int64_t b1, int32_t* st, int32_t halfI) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
    phase_roots(tid, nt, n, q, p, r, logbits, R, logp, scale, a0, b0, a1, b1);
    for (uint32_t e = tid; e < n; e += nt) { st[2 * e] = halfI; st[2 * e + 1] = 0; }
}

__global__ void k_fill_medium(Geom g, SideFB fb, BucketSet bs, uint32_t j0, uint32_t nrows, uint64_t items) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t it = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; it < items; it += stride)
        fill_medium_item(it, g, fb, bs, j0, nrows);
}

__global__ void k_fill_large(Geom g, SideFB fb, BucketSet bs, int32_t* st, uint32_t j0, uint32_t j1) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t e = fb.n_medium + blockIdx.x * blockDim.x + threadIdx.x; e < fb.n; e += stride)
        fill_large_one(e, g, fb, bs, st, j0, j1);
}

__global__ void k_region(RegionArgs a) {
    extern __shared__ uint32_t smem[];
    const Geom& g = a.g;
    RegionScratch rs;
    rs.S[0] = reinterpret_cast<uint8_t*>(smem);
    rs.S[1] = rs.S[0] + g.reg;
    rs.bitmap = smem + g.reg / 2;
    rs.surv = rs.bitmap + g.reg / 32;
    rs.misc = rs.surv + GS_MAXSURV;
    uint64_t rel = blockIdx.x, region = a.region0 + rel;
    uint32_t tid = threadIdx.x, nt = blockDim.x;
    const Update* b0 = a.bk[0] + rel * a.cap[0];
    const Update* b1 = a.bk[1] + rel * a.cap[1];
    uint32_t c0 = min(a.cnt[0][rel], a.cap[0]), c1 = min(a.cnt[1][rel], a.cap[1]);

    phase_zero(tid, nt, g, rs);
    __syncthreads();
    phase_small(tid, nt, g, a.fb[0], rs.S[0], region);
    phase_small(tid, nt, g, a.fb[1], rs.S[1], region);
    phase_buckets(tid, nt, rs.S[0], b0, c0, g.reg - 1);
    phase_buckets(tid, nt, rs.S[1], b1, c1, g.reg - 1);
    __syncthreads();
    phase_threshold(tid, nt, g, a.sq, rs, region);
    __syncthreads();
    if (tid == 0) phase_alloc(a.ro, rs);
    __syncthreads();
    phase_write_surv(tid, nt, g, a.ro, rs, region);
    phase_resieve(tid, nt, g, a.fb[0], 0, b0, c0, a.ro, rs, region);
    phase_resieve(tid, nt, g, a.fb[1], 1, b1, c1, a.ro, rs, region);
}

template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (n) CK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template <class T> T* dupload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

class GpuBackend : public Backend {
    struct Side {
        uint32_t *q = 0, *p = 0, *r = 0, *R = 0; float* logbits = 0; uint8_t* logp = 0; int32_t* st = 0;
        Update* bk = 0; uint32_t* cnt = 0;
        uint32_t n = 0, n_tiny = 0, n_small = 0, n_medium = 0;
    } sd_[2];
    Geom g_;
    BackendConfig cfg_;
    SlicePlan plan_;
    Survivor* surv_ = 0;
    Divisor* div_ = 0;
    uint32_t* ctr_ = 0;           // [0] nsurv, [1] ndiv, [2] lost surv, [3] lost div
    size_t smem_ = 0;
    std::string name_ = "gpu";
    std::vector<uint32_t> hcnt_;

    SideFB view(int s) const {
        SideFB v;
        v.q = sd_[s].q; v.p = sd_[s].p; v.R = sd_[s].R; v.logp = sd_[s].logp;
        v.n_tiny = sd_[s].n_tiny; v.n_small = sd_[s].n_small; v.n_medium = sd_[s].n_medium; v.n = sd_[s].n;
        return v;
    }

public:
    std::string name() const override { return name_; }

    ~GpuBackend() override {
        for (auto& s : sd_) {
            cudaFree(s.q); cudaFree(s.p); cudaFree(s.r); cudaFree(s.R); cudaFree(s.logbits);
            cudaFree(s.logp); cudaFree(s.st); cudaFree(s.bk); cudaFree(s.cnt);
        }
        cudaFree(surv_); cudaFree(div_); cudaFree(ctr_);
    }

    void init(const FBHost fb[2], const Geom& g, const BackendConfig& cfg) override {
        g_ = g; cfg_ = cfg;
        CK(cudaSetDevice(cfg.device));
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, cfg.device));
        name_ = std::string("gpu:") + prop.name;
        for (int s = 0; s < 2; ++s) {
            Side& d = sd_[s];
            d.n = (uint32_t)fb[s].size(); d.n_tiny = fb[s].n_tiny; d.n_small = fb[s].n_small; d.n_medium = fb[s].n_medium;
            d.q = dupload(fb[s].q); d.p = dupload(fb[s].p); d.r = dupload(fb[s].r); d.logbits = dupload(fb[s].logbits);
            d.R = dalloc<uint32_t>(d.n); d.logp = dalloc<uint8_t>(d.n); d.st = dalloc<int32_t>(2 * (size_t)d.n);
        }
        surv_ = dalloc<Survivor>(cfg.surv_cap);
        div_ = dalloc<Divisor>(cfg.div_cap);
        ctr_ = dalloc<uint32_t>(4);
        size_t freeb, totalb;
        CK(cudaMemGetInfo(&freeb, &totalb));
        double mem = cfg.mem_gb > 0 ? cfg.mem_gb * 1e9 : 0.6 * (double)freeb;
        plan_ = plan_slices(fb, g, cfg.bkmult, mem);
        for (int s = 0; s < 2; ++s) {
            sd_[s].bk = dalloc<Update>((size_t)plan_.regions_per_slice * plan_.cap[s]);
            sd_[s].cnt = dalloc<uint32_t>(plan_.regions_per_slice);
        }
        hcnt_.resize(plan_.regions_per_slice);
        smem_ = 2 * (size_t)g.reg + g.reg / 8 + (GS_MAXSURV + 4) * sizeof(uint32_t);
        if (smem_ > (size_t)prop.sharedMemPerBlockOptin)
            throw std::runtime_error("region does not fit in shared memory: use -logreg 15");
        CK(cudaFuncSetAttribute(k_region, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_));
        fprintf(stderr, "# %s: %d SMs, %.1f GB; slices of %u rows (%u slices), %lu regions, bucket cap %u/%u, %.2f GB buckets\n",
                name_.c_str(), prop.multiProcessorCount, totalb / 1e9, plan_.rows_per_slice, plan_.nslices,
                (unsigned long)plan_.regions_per_slice, plan_.cap[0], plan_.cap[1],
                (double)plan_.regions_per_slice * (plan_.cap[0] + plan_.cap[1]) * sizeof(Update) / 1e9);
    }

    void sieve_q(const QSetup& qs, QResult& out) override {
        const Geom& g = g_;
        out = QResult();
        for (int s = 0; s < 2; ++s) {
            Side& d = sd_[s];
            if (!d.n) continue;
            uint32_t blocks = std::min<uint32_t>((d.n + 255) / 256, 65535);
            k_roots<<<blocks, 256>>>(d.n, d.q, d.p, d.r, d.logbits, d.R, d.logp, qs.side[s].scale,
                                     qs.L.a0, qs.L.b0, qs.L.a1, qs.L.b1, d.st, (int32_t)g.halfI);
            CK(cudaGetLastError());
        }
        CK(cudaMemset(ctr_, 0, 4 * sizeof(uint32_t)));
        RegionArgs ra;
        ra.g = g;
        for (int s = 0; s < 2; ++s) { ra.sq[s] = qs.side[s]; ra.fb[s] = view(s); ra.cap[s] = plan_.cap[s];
                                      ra.bk[s] = sd_[s].bk; ra.cnt[s] = sd_[s].cnt; }
        ra.ro.surv = surv_; ra.ro.nsurv = ctr_; ra.ro.surv_cap = cfg_.surv_cap;
        ra.ro.div = div_; ra.ro.ndiv = ctr_ + 1; ra.ro.div_cap = cfg_.div_cap; ra.ro.lost = ctr_ + 2;

        for (uint32_t sl = 0; sl < plan_.nslices; ++sl) {
            uint32_t j0 = sl * plan_.rows_per_slice, j1 = std::min(g.J, j0 + plan_.rows_per_slice);
            uint64_t region0 = ((uint64_t)j0 * g.I) >> g.logreg;
            uint64_t nreg = (((uint64_t)j1 * g.I) >> g.logreg) - region0;
            for (int s = 0; s < 2; ++s) {
                Side& d = sd_[s];
                CK(cudaMemsetAsync(d.cnt, 0, nreg * sizeof(uint32_t)));
                BucketSet bs; bs.bk = d.bk; bs.cnt = d.cnt; bs.cap = plan_.cap[s]; bs.region0 = region0;
                SideFB v = view(s);
                uint64_t items = (uint64_t)(v.n_medium - v.n_small) * (j1 - j0);
                if (items) {
                    uint32_t blocks = (uint32_t)std::min<uint64_t>((items + FILL_THREADS - 1) / FILL_THREADS, 1u << 20);
                    k_fill_medium<<<blocks, FILL_THREADS>>>(g, v, bs, j0, j1 - j0, items);
                    CK(cudaGetLastError());
                }
                uint32_t nl = v.n - v.n_medium;
                if (nl) {
                    uint32_t blocks = std::min<uint32_t>((nl + FILL_THREADS - 1) / FILL_THREADS, 1u << 20);
                    k_fill_large<<<blocks, FILL_THREADS>>>(g, v, bs, d.st, j0, j1);
                    CK(cudaGetLastError());
                }
            }
            ra.region0 = region0;
            k_region<<<(uint32_t)nreg, REGION_THREADS, smem_>>>(ra);
            CK(cudaGetLastError());
            for (int s = 0; s < 2; ++s) {
                CK(cudaMemcpy(hcnt_.data(), sd_[s].cnt, nreg * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                for (uint64_t r = 0; r < nreg; ++r) {
                    uint32_t c = hcnt_[r];
                    out.updates[s] += c;
                    out.max_fill[s] = std::max(out.max_fill[s], (double)c / plan_.cap[s]);
                    if (c > plan_.cap[s]) out.lost_updates += c - plan_.cap[s];
                }
            }
        }
        uint32_t ctr[4];
        CK(cudaMemcpy(ctr, ctr_, sizeof ctr, cudaMemcpyDeviceToHost));
        uint32_t ns = std::min(ctr[0], cfg_.surv_cap), nd = std::min(ctr[1], cfg_.div_cap);
        out.surv.resize(ns); out.divs.resize(nd);
        if (ns) CK(cudaMemcpy(out.surv.data(), surv_, ns * sizeof(Survivor), cudaMemcpyDeviceToHost));
        if (nd) CK(cudaMemcpy(out.divs.data(), div_, nd * sizeof(Divisor), cudaMemcpyDeviceToHost));
        out.lost_surv = ctr[2]; out.lost_divs = ctr[3];
        std::sort(out.divs.begin(), out.divs.end(), [](const Divisor& a, const Divisor& b) {
            return (a.sid & 0x7FFFFFFFu) < (b.sid & 0x7FFFFFFFu);
        });
    }
};

}  // namespace

std::unique_ptr<Backend> make_gpu_backend() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return nullptr;
    return std::unique_ptr<Backend>(new GpuBackend());
}

bool gpu_backend_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
