# gsieve: a GPU lattice siever for the number field sieve

*Folder `270/Scripts for large-scale computations` of [JAgbanwa/RSA](https://github.com/JAgbanwa/RSA),
for factoring RSA-270 with the number field sieve.*

gsieve does the sieving stage of the general number field sieve on NVIDIA
GPUs, the stage that costs about 80% of the compute for a number like RSA-270.
It writes relations in CADO-NFS format, so the rest of the pipeline
(polynomial selection, filtering, linear algebra, square root) is CADO-NFS.
That is the same split as the RSA-260 (Cognition) and RSA-896 (Weis) records:
CADO-NFS with the expensive stage moved to GPUs.

## Status: read this first

* **Correctness is tested on CPU, not yet on a GPU.** The machine this was written
  on has no GPU. The CUDA build compiles cleanly (nvcc 12.0, sm_80 and sm_90, no
  register spills), and the GPU kernels call the very same functions as the CPU
  reference path (`src/core.h`), which is what all the tests below exercise.
  **The first thing to do on a rented GPU is `make && make check-gpu`**, which
  sieves the same special-q on CPU and GPU and requires identical survivors and
  identical relations.
* **Speed is unmeasured and untuned.** Expect the first GPU numbers to be well
  behind Cognition's tuned siever (~0.2 GPU-seconds per special-q at RSA-260
  size). The obvious bottlenecks are listed under "Next steps".
* gsieve is only the sieving stage. Polynomial selection, filtering and linear
  algebra are CADO-NFS on CPUs; the RSA-270 matrix will need a large cluster.

## Validation (CPU reference path, against CADO-NFS `las`)

All relations are checked by `tools/check_rels.py`, an independent pure-Python
checker: b > 0, gcd(a,b) = 1, every listed prime is prime and within its
large-prime bound, and the listed primes multiply to exactly |F(a,b)| on both
sides.

| Test | las | gsieve | in common | invalid |
|---|---|---|---|---|
| CADO's c60 (degree 4), CADO parameters, 83 special-q, I = 2^10 | 2,082 | 2,135 | 2,070 (99.6% of las) | 0 |
| RSA-210 polynomial (degree 6), 17 special-q near 1.7e7, I = 2^14, 3 large primes | 59 | 57 | 55 | 0 |
| RSA-220 polynomial (degree 6), 20 special-q near 1.1e9, I = 2^15, 3 large primes | 63 | 63 | 63 (identical sets) | 0 |

Other checks:

* **End to end.** c60 was factored by CADO-NFS from gsieve's relations alone (79,026
  relations, imported with `tasks.sieve.import`). CADO ran freerel, dup1/dup2,
  purge, merge, characters, block Wiedemann and sqrt, and never its own siever:
  679281857975170197210094035067 × 886336264679754531807636867013.
* **Region size.** The same RSA-210 special-q sieved with 2^12, 2^13, 2^14 and
  2^15-cell regions give identical relations. That covers I larger than, equal to
  and smaller than the region.
* **RSA-270 smoke test** with a base-m polynomial (`params/rsa270-basem.poly`) at
  special-q 1e9 and 5e10 (above 2^32): runs cleanly. That polynomial is too poor
  to yield relations, so it tests code paths, not yield.
* **ecm128** split all 1,500 test cofactors (2-3 primes of 33-40 bits, every factor
  checked) at 4-7x the speed of GMP-ECM (`make test-ecm`).
* The CPU reference path is single-threaded and unoptimised, so its timings
  say nothing about GPU speed.

## How it works

For each special-q (q, r) the lattice {(a,b) : a = r*b mod q} gets a reduced
basis (a0,b0),(a1,b1), and the sieve runs over cells (i,j), with
(a,b) = i*(a0,b0) + j*(a1,b1), -I/2 <= i < I/2, 0 <= j < J.

1. **Roots kernel**, one thread per factor-base entry: transform each root to the
   q-lattice (`lattice_root`). This includes the projective cases, which matter for
   powers of small primes.
2. **Bucket fill.** Primes above the region size push (offset, log p, p) updates
   into per-region buckets. Medium primes (region <= q < I) go row by row. Large
   primes (q >= I) walk the Franke-Kleinjung reduced basis, which visits every hit
   in order with two additions per hit. The walk position is saved between slices.
3. **Region kernel**, one CUDA block per 2^15 or 2^16 cells in shared memory:
   * sieve the small primes (the tiniest cooperatively, the rest one prime per thread);
   * apply the bucket updates with shared-memory atomics;
   * compare the estimated log norm minus the sieved logs with CADO's threshold
     (lambda*lpb, default mfb + 0.3*lpb);
   * drop cells with gcd(i,j) > 1;
   * resieve, recording which factor-base primes hit each survivor.
4. **Host cofactorisation**, in a thread pool that overlaps the next special-q:
   * exact norms with GMP, divided by the recorded primes;
   * CADO's size rule;
   * a 128-bit ECM written for this (`src/ecm128.cpp`, 4-7x faster per number than
     GMP-ECM on these cofactors), with GMP-ECM above 125 bits;
   * output in CADO's relation format.

The sieve area is processed in slices of rows sized to fit GPU memory.
Every phase in `src/core.h` is a `for (k = tid; k < n; k += nthreads)` loop. The
CPU backend calls it with one "thread", so the CPU path is a faithful reference
for the GPU.

## Build

Requirements: CUDA toolkit (11.5+ for `-arch=native`), g++, GMP (`libgmp-dev`),
GMP-ECM (`libecm-dev`), zlib. CADO-NFS is needed for the other stages.

```
git clone https://github.com/JAgbanwa/RSA.git
cd "RSA/270/Scripts for large-scale computations"     # quotes: the folder name has spaces
make                    # ./gsieve (CUDA), arch = the local GPU
make CUDA_ARCH=sm_100   # e.g. B200; sm_90 = H100, sm_80 = A100
make cpu                # ./gsieve-cpu, CPU reference build (no CUDA needed)
make check              # c60 test on CPU, all relations independently verified
make check-gpu          # CPU vs GPU on the same special-q: must say PASSED
make test-ecm           # ecm128 against GMP-ECM
```

## RSA-270 run, step by step

Settings are in `params/rsa270.env` (sieving) and `params/rsa270.cado-params`
(the CADO-NFS stages). They are a **starting point**, the RSA-260 published
parameters with one extra large-prime bit, and must be test-sieved.

```
export CADO=/path/to/cado-nfs                 # built with make
scripts/polyselect.sh                          # CPU: degree-6 polynomial -> params/rsa270.poly
scripts/makefb.sh                              # CPU: algebraic factor base -> params/rsa270.roots1.gz
make && make check-gpu                         # on the GPU machine
scripts/test_sieve.sh 8 16e9                   # rels/special-q, GPU-s/special-q, GPU-days estimate
scripts/sieve_node.sh 1000000000 2000000000    # every GPU on this node; run per node, disjoint ranges
scripts/to_cado.sh ~/rsa270-work/relations     # CADO: filtering, linear algebra, sqrt
```

Relation files, test-sieving output and CADO work directories go to
`$WORK_ROOT` (default `~/rsa270-work`). That keeps them out of the git checkout,
and CADO-NFS work directories should have no spaces in their path.

Rough scale: about 30 GPU-years of sieving at the efficiency of the published
records, about 16 billion raw relations, then a matrix of several hundred million
rows. `test_sieve.sh` prints the estimate for this code on your hardware.

## Options

`gsieve -h` lists them. The main ones are `-poly`, `-fb1` (CADO makefb output),
`-lim0/1`, `-lpb0/1`, `-mfb0/1`, `-A` or `-I`, `-q0/-q1`, `-nq`, `-sqside`,
`-out`, `-t` (CPU threads), `-backend gpu|cpu`, `-logreg 15|16`, `-lambda0/1`,
`-bkmult`, `-mem` (GB per slice) and `-v`. A special-q above 2^lpb on its side
gives relations CADO cannot use, and gsieve warns about it.

## Known limitations

* Bucket primes whose root is projective in the q-lattice are skipped (they occur
  with probability about 1/p).
* The single cell (1,0) on row j = 0 is skipped. It accounts for under 0.5% of
  relations.
* Prime-power projective roots are taken from CADO's factor-base file. The internal
  factor-base builder adds only first-power projective roots.
* Factor-base bounds must be below 2^32 and regions at most 2^16 cells.

## Next steps (performance)

1. Benchmark on a real GPU with `test_sieve.sh` and profile with Nsight Compute.
2. Bucket fill uses one global atomic per update. Warp-aggregated or
   shared-memory-staged writes are the standard fix and the likely main win.
3. 64-bit `%` in the medium-prime fill can be replaced by Barrett reduction or
   32-bit arithmetic.
4. Move cofactorisation (ecm128) onto the GPU, one cofactor per thread, if the host
   cannot keep up. Measure first with `-v` and `top`.
5. GPU polynomial selection and GPU block Wiedemann are not included. Cognition
   built both for RSA-260.

## Files

`src/core.h` shared GPU/CPU sieve logic · `src/backend_gpu.cu` CUDA kernels ·
`src/backend_cpu.cpp` CPU reference · `src/host.cpp` polynomials, roots mod p,
factor bases, special-q lattice · `src/cofac.cpp`, `src/ecm128.cpp`
cofactorisation · `src/main.cpp` driver and self-test · `tools/check_rels.py`
independent checker · `tests/` c60 data and ECM test · `params/` RSA-270
settings and a base-m test polynomial · `scripts/` run scripts.
