// test_ecm.cpp -- correctness and speed of ecm128 against GMP-ECM on typical NFS cofactors.
#include <ecm.h>
#include <gmpxx.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include "../src/ecm128.h"

static mpz_class rand_prime(gmp_randclass& r, int bits) {
    mpz_class p;
    do { p = r.get_z_bits(bits); mpz_setbit(p.get_mpz_t(), bits - 1); mpz_nextprime(p.get_mpz_t(), p.get_mpz_t()); }
    while ((int)mpz_sizeinbase(p.get_mpz_t(), 2) != bits);
    return p;
}

int main() {
    gmp_randclass r(gmp_randinit_default); r.seed(12345);
    struct Case { int k, bits; double B1; int curves; } cases[] = {{2, 33, 200, 18}, {3, 33, 200, 18}, {2, 37, 320, 24}, {3, 37, 320, 24}, {3, 40, 500, 24}};
    int bad = 0;
    for (auto& c : cases) {
        const int N = 300;
        std::vector<mpz_class> ns, ps;
        for (int i = 0; i < N; ++i) { mpz_class n = 1; for (int t = 0; t < c.k; ++t) n *= rand_prime(r, c.bits); ns.push_back(n); }
        auto t0 = std::chrono::steady_clock::now();
        int ok = 0;
        for (auto& n : ns) {
            mpz_class f;
            if (ecm128_factor(n, c.B1, c.curves, f)) {
                if (f <= 1 || f >= n || n % f != 0) { printf("BAD FACTOR\n"); ++bad; }
                else ++ok;
            }
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // GMP-ECM reference on the first 50 numbers with the same B1 schedule
        auto t1 = std::chrono::steady_clock::now();
        int ok2 = 0;
        for (int i = 0; i < 50; ++i) {
            for (int t = 0; t < c.curves; ++t) {
                ecm_params ep; ecm_init(ep); mpz_set_ui(ep->sigma, 0); ep->verbose = 0;
                mpz_set_d(ep->B2, 50 * c.B1 * (1 + 0.25 * t));
                mpz_class g, nn = ns[i];
                int res = ecm_factor(g.get_mpz_t(), nn.get_mpz_t(), c.B1 * (1 + 0.25 * t), ep);
                ecm_clear(ep);
                if (res > 0 && g != 1 && g != ns[i]) { ++ok2; break; }
            }
        }
        double dt2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        printf("%d x %d-bit primes: ecm128 found a factor in %d/%d (%.3f ms/number) | GMP-ECM %d/50 (%.3f ms/number)\n",
               c.k, c.bits, ok, N, 1e3 * dt / N, ok2, 1e3 * dt2 / 50);
    }
    printf("%s\n", bad ? "FAILED" : "OK");
    return bad;
}
