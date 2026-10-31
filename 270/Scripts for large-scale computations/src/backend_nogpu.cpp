// backend_nogpu.cpp -- linked instead of backend_gpu.cu in CPU-only builds.
#include "backend.h"

std::unique_ptr<Backend> make_gpu_backend() { return nullptr; }
bool gpu_backend_available() { return false; }
