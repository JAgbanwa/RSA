// main.cpp -- gsieve: special-q lattice siever for the number field sieve.
// Output: CADO-NFS relations (a,b:rational primes:algebraic primes, primes in hex).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "backend.h"
#include "cofac.h"
#include "host.h"

namespace {

struct Args {
    std::string poly, fb[2], out, backend = "auto";
    uint64_t lim[2] = {0, 0}, q0 = 0, q1 = 0;
    int lpb[2] = {0, 0}, mfb[2] = {0, 0};
    int logI = 0, A = 0, sqside = 1, threads = 0, logreg = 15, ecm_effort = 1, verbose = 0;
    long nq = -1;
    double slack = 0.0, bkmult = 1.25, mem_gb = 0, target = 200.0, lambda[2] = {0, 0};
    uint32_t powlim = 0;
    bool selftest = false;
};

[[noreturn]] void usage() {
    fprintf(stderr,
        "usage: gsieve -poly F.poly -lim0 L0 -lim1 L1 -lpb0 B0 -lpb1 B1 -mfb0 M0 -mfb1 M1\n"
        "              (-I logI | -A logArea) -q0 Q0 -q1 Q1 [options]\n"
        "  -fb1 FILE      CADO makefb factor base for side 1 (default: computed here)\n"
        "  -fb0 FILE      factor base for side 0 (default: computed here)\n"
        "  -sqside S      special-q side (default 1, algebraic)\n"
        "  -nq N          stop after N special-q\n"
        "  -out FILE      relations output (default stdout)\n"
        "  -t N           cofactorisation threads (default: all cores)\n"
        "  -backend B     gpu | cpu | auto (default auto)\n"
        "  -logreg R      log2 region size, 15 or 16 (default 15)\n"
        "  -lambda0/1 L   survivor threshold lambda*lpb bits (default mfb/lpb + 0.3, as las)\n"
        "  -slack BITS    extra threshold bits (default 0)\n"
        "  -bkmult X      bucket capacity factor (default 1.25)\n"
        "  -mem GB        bucket memory per slice (default: GPU 60%% of free, CPU 2)\n"
        "  -ecm-effort K  multiply the ECM curve budget (default 1)\n"
        "  -selftest      run CPU and GPU on the same special-q and compare\n"
        "  -v             verbose per-q statistics\n");
    exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-h" || k == "--help") usage();
        else if (k == "-poly") a.poly = need(i);
        else if (k == "-fb0") a.fb[0] = need(i);
        else if (k == "-fb1") a.fb[1] = need(i);
        else if (k == "-lim0") a.lim[0] = strtoull(need(i), 0, 10);
        else if (k == "-lim1") a.lim[1] = strtoull(need(i), 0, 10);
        else if (k == "-lpb0") a.lpb[0] = atoi(need(i));
        else if (k == "-lpb1") a.lpb[1] = atoi(need(i));
        else if (k == "-mfb0") a.mfb[0] = atoi(need(i));
        else if (k == "-mfb1") a.mfb[1] = atoi(need(i));
        else if (k == "-I") a.logI = atoi(need(i));
        else if (k == "-A") a.A = atoi(need(i));
        else if (k == "-q0") a.q0 = (uint64_t)strtod(need(i), 0);
        else if (k == "-q1") a.q1 = (uint64_t)strtod(need(i), 0);
        else if (k == "-sqside") a.sqside = atoi(need(i));
        else if (k == "-nq") a.nq = atol(need(i));
        else if (k == "-out") a.out = need(i);
        else if (k == "-t") a.threads = atoi(need(i));
        else if (k == "-backend") a.backend = need(i);
        else if (k == "-logreg") a.logreg = atoi(need(i));
        else if (k == "-slack") a.slack = atof(need(i));
        else if (k == "-lambda0") a.lambda[0] = atof(need(i));
        else if (k == "-lambda1") a.lambda[1] = atof(need(i));
        else if (k == "-bkmult") a.bkmult = atof(need(i));
        else if (k == "-mem") a.mem_gb = atof(need(i));
        else if (k == "-powlim") a.powlim = (uint32_t)atol(need(i));
        else if (k == "-ecm-effort") a.ecm_effort = atoi(need(i));
        else if (k == "-selftest") a.selftest = true;
        else if (k == "-v") a.verbose = 1;
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); usage(); }
    }
    if (a.poly.empty() || !a.lim[0] || !a.lim[1] || !a.lpb[0] || !a.lpb[1] || !a.mfb[0] || !a.mfb[1]
        || (!a.logI && !a.A) || !a.q0 || a.q1 <= a.q0)
        usage();
    if (a.lim[0] >= (1ull << 32) || a.lim[1] >= (1ull << 32)) { fprintf(stderr, "lim must be < 2^32\n"); exit(2); }
    if (a.sqside != 0 && a.sqside != 1) usage();
    return a;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// -------------------------------------------------------------- output + pool
class Output {
    FILE* f_;
    std::mutex m_;
public:
    std::atomic<uint64_t> nrels{0};
    explicit Output(const std::string& path) {
        f_ = path.empty() ? stdout : fopen(path.c_str(), "w");
        if (!f_) { perror(path.c_str()); exit(1); }
    }
    ~Output() { if (f_ != stdout) fclose(f_); }
    void lines(const std::vector<std::string>& v) {
        if (v.empty()) return;
        std::lock_guard<std::mutex> lk(m_);
        for (auto& s : v) { fputs(s.c_str(), f_); fputc('\n', f_); }
        nrels += v.size();
    }
    void comment(const std::string& s) { std::lock_guard<std::mutex> lk(m_); fprintf(f_, "%s\n", s.c_str()); fflush(f_); }
};

class Pool {
    std::vector<std::thread> th_;
    std::deque<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    size_t busy_ = 0;
    bool stop_ = false;
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                    if (q_.empty()) return;
                    job = std::move(q_.front()); q_.pop_front(); ++busy_;
                }
                job();
                { std::lock_guard<std::mutex> lk(m_); --busy_; }
                done_.notify_all();
            }
        });
    }
    ~Pool() { { std::lock_guard<std::mutex> lk(m_); stop_ = true; } cv_.notify_all(); for (auto& t : th_) t.join(); }
    void push(std::function<void()> f) { { std::lock_guard<std::mutex> lk(m_); q_.push_back(std::move(f)); } cv_.notify_one(); }
    size_t pending() { std::lock_guard<std::mutex> lk(m_); return q_.size() + busy_; }
    void wait_below(size_t n) { std::unique_lock<std::mutex> lk(m_); done_.wait(lk, [&] { return q_.size() + busy_ <= n; }); }
};

struct Totals {
    std::mutex m;
    CofacStats st;
    void add(const CofacStats& s) {
        std::lock_guard<std::mutex> lk(m);
        st.tested += s.tested; st.bad_ab += s.bad_ab; st.relations += s.relations;
        st.rejected[0] += s.rejected[0]; st.rejected[1] += s.rejected[1];
    }
};

// Special-q setup shared by the normal run and the self-test.
bool setup_q(const Poly& P, const Args& a, const Geom& g, uint64_t q, uint64_t r, QSetup& qs) {
    qs.g = g; qs.q = q; qs.r = (uint32_t)r;
    if (!reduce_qlattice(q, r, P.skew, qs.L)) return false;
    for (int s = 0; s < 2; ++s) {
        double lam = a.lambda[s] > 0 ? a.lambda[s] : (double)a.mfb[s] / a.lpb[s] + 0.3;
        make_sideq(P.f[s], qs.L, g, s == a.sqside, q, lam * a.lpb[s] + a.slack, a.target, qs.side[s]);
    }
    return true;
}

std::vector<std::pair<uint64_t, uint64_t>> special_qs(const Poly& P, const Args& a) {
    std::vector<std::pair<uint64_t, uint64_t>> v;
    mpz_class c;
    for (uint64_t q = a.q0; q < a.q1; ++q) {
        c = (unsigned long)q;
        if (!mpz_probab_prime_p(c.get_mpz_t(), 25)) continue;
        for (uint64_t r : roots_mod_p(P.f[a.sqside], q)) {
            v.push_back({q, r});
            if (a.nq >= 0 && (long)v.size() >= a.nq) return v;
        }
    }
    return v;
}

std::vector<std::string> cofactor_all(const Poly& P, const QSetup& qs, const CofacParams& cp,
                                      const QResult& res, CofacStats& st) {
    std::vector<std::string> rels;
    cofactor_range(P, qs.L, qs.q, cp, res.surv, res.divs, 0, res.surv.size(), rels, st);
    std::sort(rels.begin(), rels.end());
    return rels;
}

int run_selftest(const Poly& P, const Args& a, const Geom& g, const FBHost fb[2], const BackendConfig& bc,
                 const CofacParams& cp) {
    if (!gpu_backend_available()) { fprintf(stderr, "selftest: this build has no CUDA backend\n"); return 1; }
    auto cpu = make_cpu_backend(), gpu = make_gpu_backend();
    BackendConfig cb = bc; cb.mem_gb = bc.mem_gb > 0 ? bc.mem_gb : 2.0;
    cpu->init(fb, g, cb);
    gpu->init(fb, g, bc);
    auto qv = special_qs(P, a);
    if (qv.empty()) { fprintf(stderr, "selftest: no special-q in range\n"); return 1; }
    int fails = 0;
    for (auto& qr : qv) {
        QSetup qs;
        if (!setup_q(P, a, g, qr.first, qr.second, qs)) continue;
        QResult rc, rg;
        double t0 = now(); cpu->sieve_q(qs, rc); double t1 = now(); gpu->sieve_q(qs, rg); double t2 = now();
        auto key = [](const std::vector<Survivor>& v) {
            std::vector<std::pair<int, int>> k; for (auto& s : v) k.push_back({s.j, s.i});
            std::sort(k.begin(), k.end()); return k;
        };
        auto k1 = key(rc.surv), k2 = key(rg.surv);
        bool same_s = k1 == k2;
        CofacStats s1, s2;
        auto r1 = cofactor_all(P, qs, cp, rc, s1), r2 = cofactor_all(P, qs, cp, rg, s2);
        bool same_r = r1 == r2;
        if (!same_s || !same_r) {
            std::vector<std::pair<int, int>> d1, d2;
            std::set_difference(k1.begin(), k1.end(), k2.begin(), k2.end(), std::back_inserter(d1));
            std::set_difference(k2.begin(), k2.end(), k1.begin(), k1.end(), std::back_inserter(d2));
            printf("  survivors only on cpu %zu, only on gpu %zu", d1.size(), d2.size());
            for (size_t t = 0; t < std::min<size_t>(3, d1.size()); ++t) printf(" | cpu-only (i,j)=(%d,%d)", d1[t].second, d1[t].first);
            for (size_t t = 0; t < std::min<size_t>(3, d2.size()); ++t) printf(" | gpu-only (i,j)=(%d,%d)", d2[t].second, d2[t].first);
            printf("\n");
        }
        printf("selftest q=%lu r=%lu: cpu %zu surv %zu rels %.2fs | gpu %zu surv %zu rels %.3fs | lost cpu %lu gpu %lu | %s\n",
               (unsigned long)qs.q, (unsigned long)qs.r, rc.surv.size(), r1.size(), t1 - t0,
               rg.surv.size(), r2.size(), t2 - t1, (unsigned long)(rc.lost_updates + rc.lost_surv),
               (unsigned long)(rg.lost_updates + rg.lost_surv),
               same_s && same_r ? "MATCH" : "MISMATCH");
        if (!(same_s && same_r)) ++fails;
    }
    printf("selftest %s (%d mismatching special-q)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    double t_start = now();
    Poly P = read_poly(a.poly);

    Geom g;
    g.logI = a.logI ? a.logI : (a.A + 1) / 2;
    uint32_t logJ = a.A ? a.A - g.logI : g.logI - 1;
    g.I = 1u << g.logI; g.halfI = g.I / 2; g.J = 1u << logJ;
    g.logreg = std::min<uint32_t>(a.logreg, g.logI + logJ);
    if (g.logreg > 16) { fprintf(stderr, "logreg must be <= 16\n"); return 2; }
    g.reg = 1u << g.logreg;
    uint32_t powlim = a.powlim ? a.powlim : g.reg;

    FBHost fb[2];
    for (int s = 0; s < 2; ++s) {
        double t0 = now();
        if (!a.fb[s].empty()) load_fb_cado(a.fb[s], (uint32_t)a.lim[s], powlim, fb[s]);
        else build_fb_internal(P.f[s], (uint32_t)a.lim[s], powlim, fb[s]);
        fb[s].sort_and_split(g.reg, g.I);
        fprintf(stderr, "# side %d factor base: %zu entries (tiny %u, small %u, medium %u) in %.1fs\n", s,
                fb[s].size(), fb[s].n_tiny, fb[s].n_small - fb[s].n_tiny, fb[s].n_medium - fb[s].n_small, now() - t0);
    }

    CofacParams cp;
    for (int s = 0; s < 2; ++s) { cp.lim[s] = a.lim[s]; cp.lpb[s] = a.lpb[s]; cp.mfb[s] = a.mfb[s]; }
    cp.sqside = a.sqside; cp.ecm_effort = a.ecm_effort;
    if (a.q1 > (1ull << a.lpb[a.sqside]))
        fprintf(stderr, "# warning: special-q above 2^lpb%d will give relations CADO cannot use\n", a.sqside);

    BackendConfig bc;
    bc.bkmult = a.bkmult; bc.mem_gb = a.mem_gb;
    if (a.selftest) return run_selftest(P, a, g, fb, bc, cp);

    std::unique_ptr<Backend> be;
    if (a.backend == "gpu" || (a.backend == "auto" && gpu_backend_available())) be = make_gpu_backend();
    if (!be) {
        if (a.backend == "gpu") { fprintf(stderr, "no CUDA backend available\n"); return 1; }
        be = make_cpu_backend();
    }
    be->init(fb, g, bc);

    int nthreads = a.threads > 0 ? a.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    Output out(a.out);
    Totals tot;
    {
        char buf[512];
        snprintf(buf, sizeof buf, "# gsieve backend=%s I=2^%u J=2^%u region=2^%u lim0=%lu lim1=%lu lpb=%d,%d mfb=%d,%d sqside=%d q=[%lu,%lu)",
                 be->name().c_str(), g.logI, logJ, g.logreg, (unsigned long)a.lim[0], (unsigned long)a.lim[1],
                 a.lpb[0], a.lpb[1], a.mfb[0], a.mfb[1], a.sqside, (unsigned long)a.q0, (unsigned long)a.q1);
        out.comment(buf);
    }
    uint64_t nq = 0, nsurv = 0, lost = 0;
    double t_sieve = 0;
    {
        Pool pool(nthreads);
        mpz_class c;
        for (uint64_t q = a.q0; q < a.q1 && (a.nq < 0 || (long)nq < a.nq); ++q) {
            c = (unsigned long)q;
            if (!mpz_probab_prime_p(c.get_mpz_t(), 25)) continue;
            for (uint64_t r : roots_mod_p(P.f[a.sqside], q)) {
                if (a.nq >= 0 && (long)nq >= a.nq) break;
                auto qs = std::make_shared<QSetup>();
                if (!setup_q(P, a, g, q, r, *qs)) { fprintf(stderr, "# skip q=%lu r=%lu (lattice)\n", (unsigned long)q, (unsigned long)r); continue; }
                auto res = std::make_shared<QResult>();
                double t0 = now();
                be->sieve_q(*qs, *res);
                t_sieve += now() - t0;
                ++nq; nsurv += res->surv.size();
                lost += res->lost_updates + res->lost_surv + res->lost_divs;
                char buf[400];
                snprintf(buf, sizeof buf, "# Sieving side-%d q=%lu; rho=%lu; a0=%ld; b0=%ld; a1=%ld; b1=%ld",
                         a.sqside, (unsigned long)q, (unsigned long)r, (long)qs->L.a0, (long)qs->L.b0,
                         (long)qs->L.a1, (long)qs->L.b1);
                out.comment(buf);
                if (a.verbose)
                    fprintf(stderr, "# q=%lu r=%lu: %zu survivors, updates %lu/%lu, fill %.2f/%.2f, lost %lu/%lu/%lu, %.3fs\n",
                            (unsigned long)q, (unsigned long)r, res->surv.size(), (unsigned long)res->updates[0],
                            (unsigned long)res->updates[1], res->max_fill[0], res->max_fill[1],
                            (unsigned long)res->lost_updates, (unsigned long)res->lost_surv,
                            (unsigned long)res->lost_divs, now() - t0);
                // cofactorise in chunks on the pool while the backend moves on
                const size_t CH = 32;
                for (size_t b = 0; b < res->surv.size(); b += CH) {
                    size_t e = std::min(res->surv.size(), b + CH);
                    pool.push([&P, &cp, &out, &tot, qs, res, b, e] {
                        std::vector<std::string> rels; CofacStats st;
                        cofactor_range(P, qs->L, qs->q, cp, res->surv, res->divs, b, e, rels, st);
                        out.lines(rels); tot.add(st);
                    });
                }
                pool.wait_below((size_t)nthreads * 64);
            }
        }
        pool.wait_below(0);
    }
    double el = now() - t_start;
    char buf[400];
    snprintf(buf, sizeof buf, "# %lu special-q, %lu survivors, %lu tested, rejected %lu/%lu, bad (a,b) %lu, lost %lu",
             (unsigned long)nq, (unsigned long)nsurv, (unsigned long)tot.st.tested,
             (unsigned long)tot.st.rejected[0], (unsigned long)tot.st.rejected[1],
             (unsigned long)tot.st.bad_ab, (unsigned long)lost);
    out.comment(buf);
    snprintf(buf, sizeof buf, "# Total sieve time %.2fs (%.4fs/special-q)", t_sieve, nq ? t_sieve / nq : 0.0);
    out.comment(buf);
    snprintf(buf, sizeof buf, "# Total elapsed time %.2fs", el);
    out.comment(buf);
    uint64_t nr = out.nrels.load();
    snprintf(buf, sizeof buf, "# Total %lu reports [%.3gs/r, %.1fr/sq]", (unsigned long)nr,
             nr ? el / nr : 0.0, nq ? (double)nr / nq : 0.0);
    out.comment(buf);
    return 0;
}
