// backend_cpu.cpp -- reference backend: runs the core.h phases with one "thread".
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "backend.h"

namespace {
class CpuBackend : public Backend {
    FBHost fb_[2];
    Geom g_;
    BackendConfig cfg_;
    SlicePlan plan_;
    std::vector<uint32_t> R_[2];
    std::vector<uint8_t> logp_[2];
    std::vector<int32_t> state_[2];
    std::vector<Update> bk_[2];
    std::vector<uint32_t> cnt_[2];
    std::vector<Survivor> surv_buf_;
    std::vector<Divisor> div_buf_;
    // GS_DEBUG_CELL="q,i,j": print sieve values of one cell (diagnostics only)
    bool dbg_ = false;
    uint64_t dq_ = 0; int32_t di_ = 0, dj_ = 0;

    void debug_cell(const QSetup& qs, const RegionScratch& rs, uint64_t region) {
        if (qs.q != dq_) return;
        uint64_t x = (uint64_t)dj_ * g_.I + (uint64_t)(di_ + (int32_t)g_.halfI);
        if ((x >> g_.logreg) != region) return;
        uint32_t c = (uint32_t)(x & (g_.reg - 1));
        float xx = (float)di_ / g_.halfI, yy = (float)dj_ / g_.J;
        for (int s = 0; s < 2; ++s) {
            float L = lognorm_units(qs.side[s], xx, yy);
            fprintf(stderr, "# debug q=%lu (%d,%d) side %d: lognorm %.1f units (%.2f bits), sieved %u units (%.2f bits), "
                    "residual %.2f bits, thresh %.2f bits\n", (unsigned long)qs.q, di_, dj_, s, L, L / qs.side[s].scale,
                    rs.S[s][c], rs.S[s][c] / qs.side[s].scale, (L - rs.S[s][c]) / qs.side[s].scale,
                    qs.side[s].thresh / qs.side[s].scale);
        }
    }

public:
    std::string name() const override { return "cpu"; }

    void init(const FBHost fb[2], const Geom& g, const BackendConfig& cfg) override {
        g_ = g; cfg_ = cfg;
        if (const char* d = getenv("GS_DEBUG_CELL")) {
            unsigned long q; int i, j;
            if (sscanf(d, "%lu,%d,%d", &q, &i, &j) == 3) { dbg_ = true; dq_ = q; di_ = i; dj_ = j; }
        }
        for (int s = 0; s < 2; ++s) {
            fb_[s] = fb[s];
            R_[s].resize(fb_[s].size());
            logp_[s].resize(fb_[s].size());
            state_[s].resize(2 * fb_[s].size());
        }
        double mem = cfg.mem_gb > 0 ? cfg.mem_gb * 1e9 : 2e9;
        plan_ = plan_slices(fb_, g_, cfg.bkmult, mem);
        for (int s = 0; s < 2; ++s) {
            bk_[s].resize((size_t)plan_.regions_per_slice * plan_.cap[s]);
            cnt_[s].resize(plan_.regions_per_slice);
        }
    }

    SideFB view(int s) const {
        SideFB v;
        v.q = fb_[s].q.data(); v.p = fb_[s].p.data(); v.R = R_[s].data(); v.logp = logp_[s].data();
        v.n_tiny = fb_[s].n_tiny; v.n_small = fb_[s].n_small; v.n_medium = fb_[s].n_medium;
        v.n = (uint32_t)fb_[s].size();
        return v;
    }

    void sieve_q(const QSetup& qs, QResult& out) override {
        const Geom& g = g_;
        out = QResult();
        for (int s = 0; s < 2; ++s) {
            phase_roots(0, 1, (uint32_t)fb_[s].size(), fb_[s].q.data(), fb_[s].p.data(), fb_[s].r.data(),
                        fb_[s].logbits.data(), R_[s].data(), logp_[s].data(), qs.side[s].scale,
                        qs.L.a0, qs.L.b0, qs.L.a1, qs.L.b1);
            for (size_t e = 0; e < fb_[s].size(); ++e) { state_[s][2 * e] = (int32_t)g.halfI; state_[s][2 * e + 1] = 0; }
        }
        if (surv_buf_.size() != cfg_.surv_cap) surv_buf_.resize(cfg_.surv_cap);
        if (div_buf_.size() != cfg_.div_cap) div_buf_.resize(cfg_.div_cap);
        uint32_t nsurv = 0, ndiv = 0, lost[2] = {0, 0};
        RegionOut ro;
        ro.surv = surv_buf_.data(); ro.nsurv = &nsurv; ro.surv_cap = cfg_.surv_cap;
        ro.div = div_buf_.data(); ro.ndiv = &ndiv; ro.div_cap = cfg_.div_cap; ro.lost = lost;

        std::vector<uint32_t> scratch(g.reg / 4 * 2 + g.reg / 32 + GS_MAXSURV + 4);
        RegionScratch rs;
        rs.S[0] = reinterpret_cast<uint8_t*>(scratch.data());
        rs.S[1] = rs.S[0] + g.reg;
        rs.bitmap = scratch.data() + g.reg / 2;
        rs.surv = rs.bitmap + g.reg / 32;
        rs.misc = rs.surv + GS_MAXSURV;

        uint32_t rows_per_region = g.logI >= g.logreg ? 1 : (g.reg >> g.logI);
        for (uint32_t sl = 0; sl < plan_.nslices; ++sl) {
            uint32_t j0 = sl * plan_.rows_per_slice;
            uint32_t j1 = std::min(g.J, j0 + plan_.rows_per_slice);
            uint64_t region0 = ((uint64_t)j0 * g.I) >> g.logreg;
            uint64_t nreg = (((uint64_t)j1 * g.I) >> g.logreg) - region0;
            if (g.logI < g.logreg) nreg = (j1 - j0) / rows_per_region;
            BucketSet bs[2];
            for (int s = 0; s < 2; ++s) {
                std::fill(cnt_[s].begin(), cnt_[s].end(), 0u);
                bs[s].bk = bk_[s].data(); bs[s].cnt = cnt_[s].data(); bs[s].cap = plan_.cap[s]; bs[s].region0 = region0;
                SideFB v = view(s);
                uint64_t nmed = v.n_medium - v.n_small, items = nmed * (j1 - j0);
                for (uint64_t it = 0; it < items; ++it) fill_medium_item(it, g, v, bs[s], j0, j1 - j0);
                for (uint32_t e = v.n_medium; e < v.n; ++e) fill_large_one(e, g, v, bs[s], state_[s].data(), j0, j1);
                for (uint64_t r = 0; r < nreg; ++r) {
                    uint32_t c = cnt_[s][r];
                    out.updates[s] += c;
                    out.max_fill[s] = std::max(out.max_fill[s], (double)c / plan_.cap[s]);
                    if (c > plan_.cap[s]) out.lost_updates += c - plan_.cap[s];
                }
            }
            SideFB v0 = view(0), v1 = view(1);
            for (uint64_t r = 0; r < nreg; ++r) {
                uint64_t region = region0 + r;
                uint32_t c0 = std::min(cnt_[0][r], plan_.cap[0]), c1 = std::min(cnt_[1][r], plan_.cap[1]);
                const Update* b0 = bk_[0].data() + r * plan_.cap[0];
                const Update* b1 = bk_[1].data() + r * plan_.cap[1];
                phase_zero(0, 1, g, rs);
                phase_small(0, 1, g, v0, rs.S[0], region);
                phase_small(0, 1, g, v1, rs.S[1], region);
                phase_buckets(0, 1, rs.S[0], b0, c0, g.reg - 1);
                phase_buckets(0, 1, rs.S[1], b1, c1, g.reg - 1);
                if (dbg_) debug_cell(qs, rs, region);
                phase_threshold(0, 1, g, qs.side, rs, region);
                phase_alloc(ro, rs);
                phase_write_surv(0, 1, g, ro, rs, region);
                phase_resieve(0, 1, g, v0, 0, b0, c0, ro, rs, region);
                phase_resieve(0, 1, g, v1, 1, b1, c1, ro, rs, region);
            }
        }
        out.surv.assign(surv_buf_.begin(), surv_buf_.begin() + std::min(nsurv, cfg_.surv_cap));
        out.divs.assign(div_buf_.begin(), div_buf_.begin() + std::min(ndiv, cfg_.div_cap));
        out.lost_surv = lost[0]; out.lost_divs = lost[1];
        std::sort(out.divs.begin(), out.divs.end(), [](const Divisor& a, const Divisor& b) {
            return (a.sid & 0x7FFFFFFFu) < (b.sid & 0x7FFFFFFFu);
        });
    }
};
}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::unique_ptr<Backend>(new CpuBackend()); }
