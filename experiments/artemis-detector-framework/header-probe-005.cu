#include <ATen/cuda/CUDAContext.h>
// Compile-only inclusion check; no launch, detector, weights or computation.
__global__ void paper6_header_inclusion_probe() {}
