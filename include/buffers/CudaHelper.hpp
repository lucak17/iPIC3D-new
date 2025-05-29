#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                  \
  do {                                                    \
    cudaError_t err = call;                               \
    if (err != cudaSuccess)                               \
      throw std::runtime_error(cudaGetErrorString(err));  \
  } while (0)
#endif
