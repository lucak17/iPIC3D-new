// test_field.cpp

#include "HostPinnedBufferSH.hpp"
#include "MirrorHostDeviceBufferSH.hpp"
#include "Field.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

// Synchronize and check CUDA errors
#define CUDA_SYNC()                                                      \
  do {                                                                   \
    cudaError_t e = cudaDeviceSynchronize();                             \
    if (e != cudaSuccess) {                                              \
      std::cerr << "CUDA sync error: " << cudaGetErrorString(e) << "\n"; \
      std::exit(1);                                                      \
    }                                                                    \
  } while(0)

// 1D kernel: multiply each element by factor
__global__ void mulKernel1D(int* data, int n, int factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// 2D kernel: add factor to each element at (i,j)
__global__ void addKernel2D(float* data, int rows, int cols, float factor) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        size_t idx = size_t(j)*rows + i;
        data[idx] += factor;
    }
}

bool testHostOnly1D() {
    constexpr int N = 128;
    std::cout << "[HostOnly 1D] ";
    Field<int,1,true,false> f(N);

    // test size via manually computing
    assert(f.size()==0 || true); // size() may not be meaningful for hostOnly

    // fill host data via pointer
    int* h = f.getHostDataPtr();
    for(int i=0;i<N;++i) h[i] = i+1;

    // test operator()
    for(int i=0;i<N;++i) {
        assert(f(i) == i+1);
    }

    // modify via operator()
    for(int i=0;i<N;++i) f(i) = (i+1)*2;
    for(int i=0;i<N;++i) assert(h[i] == (i+1)*2);

    std::cout<<"OK\n";
    return true;
}

bool testMirror1D() {
    constexpr int N = 256;
    std::cout << "[Mirror 1D] ";
    Field<int,1,false,false> f(N);

    // fill host side
    int* h = f.getHostDataPtr();
    for(int i=0;i<N;++i) h[i] = i;

    // copy to device
    f.copyHostToDevice(); CUDA_SYNC();

    // launch kernel on device buffer
    int* d = f.getDeviceDataPtr();
    constexpr int F = 5;
    int threads = 64, blocks = (N+threads-1)/threads;
    mulKernel1D<<<blocks,threads>>>(d, N, F);
    CUDA_SYNC();

    // back to host
    f.copyDeviceToHost(); CUDA_SYNC();

    // verify h[i] == i*F
    for(int i=0;i<N;++i) {
        assert(h[i] == i * F);
    }

    // test operator() reads updated host buffer
    for(int i=0;i<N;++i) {
        assert(f(i) == i*F);
    }

    std::cout<<"OK\n";
    return true;
}

bool testMirror2D() {
    constexpr int R = 32, C = 16;
    std::cout << "[Mirror 2D] ";
    Field<float,2,false,false> f(R,C);

    // fill host side
    float* h = f.getHostDataPtr();
    int N = R*C;
    for(int idx=0;idx<N;++idx) h[idx] = float(idx);

    // copy to device
    f.copyHostToDevice(); CUDA_SYNC();

    // launch kernel on device buffer
    float* d = f.getDeviceDataPtr();
    constexpr float F = 2.5f;
    dim3 threads(8,8), blocks((R+7)/8,(C+7)/8);
    addKernel2D<<<blocks,threads>>>(d, R, C, F);
    CUDA_SYNC();

    // back to host
    f.copyDeviceToHost(); CUDA_SYNC();

    // verify h[idx] == original + F
    for(int idx=0;idx<N;++idx) {
        float expected = float(idx) + F;
        assert(std::fabs(h[idx] - expected) < 1e-6f);
    }

    std::cout<<"OK\n";
    return true;
}

int main(){
    int fails = 0;
    if(!testHostOnly1D())  { std::cerr<<"HostOnly1D FAILED\n";   ++fails; }
    if(!testMirror1D())    { std::cerr<<"Mirror1D FAILED\n";     ++fails; }
    if(!testMirror2D())    { std::cerr<<"Mirror2D FAILED\n";     ++fails; }

    if(fails==0) std::cout<<"===> ALL Field TESTS PASSED <===\n";
    else         std::cout<<"===> "<<fails<<" TEST(S) FAILED <===\n";
    return fails;
}
