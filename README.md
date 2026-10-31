# RSA

Work on factoring numbers from the RSA Factoring Challenge with the general
number field sieve (GNFS).

## RSA-270

RSA-270 is a 270-digit, 895-bit semiprime from the challenge list.

```
2331085303444075445276376569106805241456198124803054490429486119684959182451
3578286788836931857711641821391926857265831491306067262691135402760979316634
1626693946596196427744273886601876896313468704059066746903123910748277606548
649151920812699309766587514735456594993207
```

As of October 2026 no factorisation has been published. Its neighbours on the list
have both just fallen:

* RSA-260 (862 bits), on 3 September 2026, by Eric Lu (Cognition);
* RSA-896 (896 bits), on 19 September 2026, by Stephen Weis (Anthropic).

Both runs used CADO-NFS with the sieving stage ported to GPUs. RSA-270 needs
about the same work as RSA-896: roughly 30 GPU-years of sieving at those runs'
efficiency.

### Approach

This uses the same split as those records:

* **CADO-NFS** does polynomial selection, factor bases, filtering, linear algebra
  and the square root.
* **gsieve**, a GPU lattice siever written for this, does the sieving, which is
  about 80% of the compute. It writes CADO-NFS relations.

### [`270/Scripts for large-scale computations`](270/Scripts%20for%20large-scale%20computations)

| Path | What it is |
|---|---|
| `src/` | gsieve: CUDA kernels (`backend_gpu.cu`), sieve logic shared with a CPU reference (`core.h`), factor bases and special-q lattice (`host.cpp`), cofactorisation with a 128-bit ECM (`cofac.cpp`, `ecm128.cpp`) |
| `params/rsa270.env` | sieving settings for RSA-270 (starting point, to be test-sieved) |
| `params/rsa270.cado-params` | CADO-NFS settings for the other stages |
| `params/rsa270-basem.poly` | a valid but poor base-m polynomial, for smoke tests only |
| `scripts/` | polynomial selection, factor base, test sieving, multi-GPU sieving, hand-off to CADO-NFS |
| `tools/check_rels.py` | independent relation checker |
| `tests/` | c60 test data, ECM test |

See that folder's [README](270/Scripts%20for%20large-scale%20computations/README.md)
for the design, the validation results, the build and the step-by-step run.

### Status

* The siever is written and checked against CADO-NFS's own siever `las`, all on the
  CPU reference path. On the RSA-220 polynomial at special-q near 1.1e9 it finds
  the same 63 relations as `las`. CADO-NFS factored a 60-digit test number from
  gsieve's relations alone.
* **It has not run on a GPU yet.** The CUDA build compiles for A100 (sm_80) and
  H100 (sm_90). The first step on a GPU is
  `make && make check-gpu`, which compares the GPU and CPU paths on the same
  special-q.
* GPU speed is not measured or tuned yet.
* No RSA-270 polynomial yet. `scripts/polyselect.sh` runs CADO-NFS polynomial
  selection, a large CPU job.
* Linear algebra will need a large cluster: RSA-260's matrix had 656M rows.

### Quick start

```
git clone https://github.com/JAgbanwa/RSA.git
cd "RSA/270/Scripts for large-scale computations"
make cpu && make check     # CPU reference build and c60 test, no GPU needed
make && make check-gpu     # on a machine with an NVIDIA GPU
```
