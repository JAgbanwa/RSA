# gsieve -- GPU lattice siever for the number field sieve.
#
#   make            CUDA build (needs nvcc), binary ./gsieve
#   make cpu        CPU-only reference build, binary ./gsieve-cpu
#
# Variables: CUDA_ARCH (default native, e.g. sm_90 for H100, sm_100 for B200),
#            NVCC, CXX, CCBIN (host compiler handed to nvcc).

CXX       ?= g++
NVCC      ?= nvcc
CUDA_ARCH ?= native
CCBIN     ?= $(CXX)
CXXFLAGS  ?= -O3 -march=native
CXXFLAGS  += -std=c++17 -Wall -Wextra -Wno-unused-parameter -fno-strict-aliasing -ffp-contract=off -pthread
NVFLAGS   ?= -O3
NVFLAGS   += -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(CCBIN) --fmad=false -Xcompiler -fno-strict-aliasing,-ffp-contract=off,-pthread
LIBS       = -lecm -lgmpxx -lgmp -lz -lpthread

HDRS = src/core.h src/host.h src/cofac.h src/ecm128.h src/backend.h
SRCS = src/main.cpp src/host.cpp src/cofac.cpp src/ecm128.cpp src/backend_cpu.cpp src/backend_common.cpp

all: gsieve

cpu: gsieve-cpu

build/cpu/%.o: src/%.cpp $(HDRS)
	@mkdir -p build/cpu
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/gpu/%.o: src/%.cpp $(HDRS)
	@mkdir -p build/gpu
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/gpu/backend_gpu.o: src/backend_gpu.cu $(HDRS)
	@mkdir -p build/gpu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

gsieve-cpu: $(SRCS:src/%.cpp=build/cpu/%.o) build/cpu/backend_nogpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

gsieve: $(SRCS:src/%.cpp=build/gpu/%.o) build/gpu/backend_gpu.o
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf build gsieve gsieve-cpu

.PHONY: all cpu clean check check-gpu

# Self-contained check: sieve CADO's c60 test number (factor bases built here)
# and verify every relation independently.
check: gsieve-cpu
	./gsieve-cpu -poly tests/c60.poly -lim0 78682 -lim1 111342 -lpb0 18 -lpb1 19 \
	    -mfb0 17 -mfb1 38 -I 10 -logreg 12 -q0 61961 -q1 62961 -nq 30 -out tests/c60.check.rels
	python3 tools/check_rels.py tests/c60.poly tests/c60.check.rels -lpb0 18 -lpb1 19
	@n=$$(grep -c -v '^#' tests/c60.check.rels); test $$n -gt 500 && echo "check OK: $$n valid relations"

# GPU check: CPU and GPU must produce identical survivors and relations.
check-gpu: gsieve
	./gsieve -poly tests/c60.poly -lim0 78682 -lim1 111342 -lpb0 18 -lpb1 19 \
	    -mfb0 17 -mfb1 38 -I 10 -logreg 12 -q0 61961 -q1 62961 -nq 10 -selftest

.PHONY: check check-gpu

# ecm128 correctness and speed against GMP-ECM
test-ecm: tests/test_ecm.cpp src/ecm128.cpp src/ecm128.h
	$(CXX) $(CXXFLAGS) -o tests/test_ecm tests/test_ecm.cpp src/ecm128.cpp $(LIBS)
	./tests/test_ecm

.PHONY: test-ecm
