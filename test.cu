// test.cpp

#include "hostPinnedBufferSH.hpp"
#include "hostBufferSH.hpp"
#include "deviceBufferSH.hpp"
#include "mirrorHostDeviceBufferSH.hpp"
#include "field.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Synchronize and check CUDA errors
#define CUDA_SYNC()                                                      \
  do {                                                                   \
    cudaError_t e = cudaDeviceSynchronize();                             \
    if (e != cudaSuccess) {                                              \
      std::cerr << "CUDA sync error: "                                   \
                << cudaGetErrorString(e) << "\n";                        \
      return false;                                                      \
    }                                                                    \
  } while(0)

// Kernel to fill int buffer with idx * factor
__global__ void fillKernel(int* data, size_t n, int factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = int(idx) * factor;
}

// Tolerance omitted (we only test ints here)

//------------------------------------------------------------------------------
// Comprehensive test for 1D buffers and Field, parameterized by unified
//------------------------------------------------------------------------------
template<bool unified>
bool runTests1D(size_t N) {
    std::cout << "=== Testing "
              << (unified ? "unified" : "non-unified")
              << " mode, N=" << N << " ===\n";

    std::cout<< "Starting test 1" <<std::endl;
    // 1) HostPinnedBuffer: CPU fill & host<->host copy
    {
        HostPinnedBuffer<int,1,unified> hp(N);
        std::vector<int> src(N), dst(N,0);
        for (size_t i=0;i<N;++i) src[i] = int(i*3+1);
        hp.copyFromHost(src.data(), N);
        hp.copyToHost(dst.data(), N);
        for (size_t i=0;i<N;++i) {
            if (dst[i] != src[i]) {
                std::cerr<<"HP host<->host mismatch at "<<i<<"\n";
                return false;
            }
        }
    }
    std::cout<< "Test 1 done, starting test 2" <<std::endl;
    // 2) HostBuffer indexing test
    {
        HostBuffer<int,1,false> hb(N);
        for (size_t i=0;i<N;++i) hb(i) = int(i*2);
        for (size_t i=0;i<N;++i) {
            if (hb(i) != int(i*2)) {
                std::cerr<<"HB indexing mismatch at "<<i<<"\n";
                return false;
            }
        }
    }
    std::cout<< "Test 2 done, starting test 3" <<std::endl;
    // 3) HostPinnedBuffer <-> DeviceBuffer communication
    DeviceBuffer<int,1,unified> d(N);
    {
        HostPinnedBuffer<int,1,unified> hp(N);
        for (size_t i=0;i<N;++i) hp.getDataPtr()[i] = int(i+5);

        // host -> device
        hp.copyToDevice(d.getDataPtr(), N);
        CUDA_SYNC();
        std::cout<< "Test 3 host device done" <<std::endl;
        // device -> host
        HostPinnedBuffer<int,1,unified> hp2(N);
        hp2.copyFromDevice(d.getDataPtr(), N);
        CUDA_SYNC();
        for (size_t i=0;i<N;++i) {
            if (hp2.getDataPtr()[i] != hp.getDataPtr()[i]) {
                std::cerr<<"hp->d->hp2 mismatch at "<<i<<"\n";
                return false;
            }
        }
        std::cout<< "Test 3 device host done" <<std::endl;
        // device -> device
        DeviceBuffer<int,1,unified> d2(N);
        HostBuffer<int,1,unified> h2(N);
        d.copyDeviceDevice(d2.getDataPtr(), d.getDataPtr(), N);
        h2.copyFromDevice(d2.getDataPtr(),N);
        std::cout<< "Test 3 device device done 0" <<std::endl;
        for (size_t i=0;i<N;++i) {
            if (h2.getDataPtr()[i] != hp.getDataPtr()[i]) {
                std::cerr<<"d->d2 mismatch at "<<i<<"\n";
                return false;
            }
        }
        std::cout<< "Test 3 device device done" <<std::endl;
    }

    std::cout<< "Test 3 done, starting test 4" <<std::endl;
    // 4) GPU kernel on DeviceBuffer
    {
        DeviceBuffer<int,1,unified> d(N);
        int *ptr = d.getDataPtr();
        constexpr int factor = 7;
        dim3 block(256), grid((N+block.x-1)/block.x);
        fillKernel<<<grid,block>>>(ptr, N, factor);
        CUDA_SYNC();

        // back to host
        std::vector<int> host(N);
        d.copyToHost(host.data(), N);
        CUDA_SYNC();
        for (size_t i=0;i<N;++i) {
            if (host[i] != int(i)*factor) {
                std::cerr<<"DeviceBuffer kernel mismatch at "<<i<<"\n";
                return false;
            }
        }
    }

    std::cout<< "Test 4 done, starting test 5" <<std::endl;
    // 5) MirrorHostDeviceBuffer: CPU fill, kernel, back
    {
        MirrorHostDeviceBuffer<int,1,unified> m(N);
        int *hptr = m.getHostDataPtr();
        int *dptr = m.getDeviceDataPtr();

        // CPU init
        for (size_t i=0;i<N;++i) hptr[i] = int(i*4 + 2);
        m.copyHostToDevice();
        CUDA_SYNC();

        // kernel modifies on GPU
        constexpr int factor = 5;
        dim3 block(256), grid((N+block.x-1)/block.x);
        fillKernel<<<grid,block>>>(dptr, N, factor);
        CUDA_SYNC();

        // back to CPU
        m.copyDeviceToHost();
        CUDA_SYNC();
        for (size_t i=0;i<N;++i) {
            if (hptr[i] != int(i)*factor) {
                std::cerr<<"Mirror kernel mismatch at "<<i<<"\n";
                return false;
            }
        }
    }
    std::cout<< "Test 5 done, starting test 6" <<std::endl;
    // 6) Field: CPU fill, kernel, back
    {
        Field<int,1,unified> f(N);
        int *hptr = f.getHostDataPtr();
        int *dptr = f.getDeviceDataPtr();

        // CPU fill
        for (size_t i=0;i<N;++i) hptr[i] = int(i*9 - 3);
        f.copyHostToDevice();
        CUDA_SYNC();

        // kernel
        constexpr int factor = 11;
        dim3 block(256), grid((N+block.x-1)/block.x);
        fillKernel<<<grid,block>>>(dptr, N, factor);
        CUDA_SYNC();

        // back
        f.copyDeviceToHost();
        CUDA_SYNC();
        for (size_t i=0;i<N;++i) {
            if (hptr[i] != int(i)*factor) {
                std::cerr<<"Field kernel mismatch at "<<i<<"\n";
                return false;
            }
        }
    }
    std::cout<< "Test 6 done, starting test 7" <<std::endl;
    // 7) Unified buffer aliasing test (only when unified=true)
    if constexpr (unified) {

        MirrorHostDeviceBuffer<int,1,true> mu(N);
        HostPinnedBuffer<int,1,true> hu(N);
        DeviceBuffer<int,1,true>      du(N);
        if (mu.getHostDataPtr() != mu.getDeviceDataPtr()) {
            std::cerr<<"Unified aliasing failed\n";
            return false;
        }
        // kernel directly on unified pointer
        int *ptr = hu.getDataPtr();
        constexpr int factor = 13;
        dim3 block(256), grid((N+block.x-1)/block.x);
        fillKernel<<<grid,block>>>(ptr, N, factor);
        CUDA_SYNC();
        for (size_t i=0;i<N;++i) {
            if (ptr[i] != int(i)*factor) {
                std::cerr<<"Unified kernel mismatch at "<<i<<"\n";
                return false;
            }
        }
    }

    std::cout<<"  all tests passed for N="<<N<<"\n\n";
    return true;
}


// 2D kernel: adds factor to each element with i fastest
__global__ void addKernel2D(float* data, int rows, int cols, float factor) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;  // i fastest
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        int idx = j*rows + i;   // contiguous along i
        data[idx] += factor;
    }
}

// 3D kernel: multiplies each element by factor with i fastest
__global__ void mulKernel3D(double* data, int X, int Y, int Z, double factor) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;  // i fastest
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i < X && j < Y && k < Z) {
        size_t idx = size_t(k)*X*Y + size_t(j)*X + i;
        data[idx] *= factor;
    }
}

//----------------------------------------------------------------------------
// Test MirrorHostDeviceBuffer in 2D, verifying contiguous layout & expand
//----------------------------------------------------------------------------
template<bool unified>
bool testMirror2D(int rows, int cols) {
    std::cout<<"testMirror2D<"<<unified<<">("<<rows<<","<<cols<<")\n";
    int N = rows*cols;

    MirrorHostDeviceBuffer<float,2,unified> m(rows,cols);
    float* h = m.getHostDataPtr();
    float* d = m.getDeviceDataPtr();

    // CPU initialize via pointer contiguously
    for (size_t idx=0; idx<N; ++idx) {
        // i = idx % rows, j = idx / rows
        int i = idx % rows, j = idx / rows;
        h[idx] = float(i) - float(j);
    }

    std::cout<< "Test 6 done, starting test 7" <<std::endl;
    // verify operator()(i,j) matches pointer
    for (int i=0; i<rows; ++i)
      for (int j=0; j<cols; ++j) {
        size_t idx = size_t(j)*rows + i;
        assert(m(i,j) == h[idx]);
      }
    
    std::cout<< "Test " << m.getHostBufferPtr()->size() << " " <<m.getDeviceBufferPtr()->size() << std::endl;
    // host->device, corrupt host, device->host
    m.copyHostToDevice(); CUDA_SYNC();
    std::fill(h, h+N, 0);
    m.copyDeviceToHost(); CUDA_SYNC();
    for (size_t idx=0; idx<N; ++idx) {
        int i = idx % rows, j = idx / rows;
        float expected = float(i) - float(j);
        assert(fabs(h[idx] - expected) < 1e-3f);
    }

    // GPU kernel add
    constexpr float F = 2.5f;
    dim3 block(16,16), grid((rows+15)/16,(cols+15)/16);
    addKernel2D<<<grid,block>>>(d, rows, cols, F);
    CUDA_SYNC();

    m.copyDeviceToHost(); CUDA_SYNC();
    for (size_t idx=0; idx<N; ++idx) {
        int i = idx % rows, j = idx / rows;
        float expected = (float(i) - float(j)) + F;
        assert(fabs(h[idx] - expected) < 1e-6f);
    }

    // expand to double size
    int R2 = rows*2, C2 = cols*2;
    m.expandBuffer(R2,C2);
    float* h2 = m.getHostDataPtr();
    float* d2 = m.getDeviceDataPtr();
    size_t N2 = size_t(R2)*C2;

    // old region preserved
    for (size_t idx=0; idx<N; ++idx) {
        int i = idx % rows, j = idx / rows;
        float expected = (float(i) - float(j)) + F;
        assert(fabs(h2[idx] - expected) < 1e-6f);
    }

    // initialize new region (cols..C2)
    for (int i=0;i<R2;++i)
      for (int j=cols;j<C2;++j) {
        size_t idx = size_t(j)*R2 + i;
        h2[idx] = float(100 + i + j);
      }

    m.copyHostToDevice(); CUDA_SYNC();
    addKernel2D<<<dim3((R2+15)/16,(C2+15)/16),block>>>(d2, R2, C2, F);
    CUDA_SYNC();
    m.copyDeviceToHost(); CUDA_SYNC();

    for (int i=0;i<R2;++i)
      for (int j=cols;j<C2;++j) {
        size_t idx = size_t(j)*R2 + i;
        float expected = float(100 + i + j) + F;
        assert(fabs(h2[idx] - expected) < 1e-6f);
      }

    return true;
}

//----------------------------------------------------------------------------
// Test Field in 3D, verifying contiguous layout, operator(), and kernel
//----------------------------------------------------------------------------
template<bool unified>
bool testField3D(int X, int Y, int Z) {
    std::cout<<"testField3D<"<<unified<<">("<<X<<","<<Y<<","<<Z<<")\n";
    size_t N = size_t(X)*Y*Z;
    Field<double,3,unified> f(X,Y,Z);
    double* h = f.getHostDataPtr();
    double* d = f.getDeviceDataPtr();

    // CPU initialize via operator()
    for (int i=0;i<X;++i)
      for (int j=0;j<Y;++j)
        for (int k=0;k<Z;++k)
          f(i,j,k) = i + j*10 + k*100;

    // verify contiguous pointer matches operator()
    for (int i=0;i<X;++i)
      for (int j=0;j<Y;++j)
        for (int k=0;k<Z;++k) {
          size_t idx = size_t(k)*X*Y + size_t(j)*X + i;
          assert(h[idx] == f(i,j,k));
        }

    // host->device->host round-trip
    f.copyHostToDevice(); CUDA_SYNC();
    std::fill(h, h+N, 0.0);
    f.copyDeviceToHost(); CUDA_SYNC();
    for (int i=0;i<X;++i)
      for (int j=0;j<Y;++j)
        for (int k=0;k<Z;++k)
          assert(f(i,j,k) == i + j*10 + k*100);

    // GPU kernel multiply
    constexpr double F = 1.5;
    dim3 block(8,8,4),
         grid((X+7)/8,(Y+7)/8,(Z+3)/4);
    mulKernel3D<<<grid,block>>>(d, X, Y, Z, F);
    CUDA_SYNC();

    f.copyDeviceToHost(); CUDA_SYNC();
    for (int i=0;i<X;++i)
      for (int j=0;j<Y;++j)
        for (int k=0;k<Z;++k) {
          double expected = (i + j*10 + k*100)*F;
          assert(fabs(f(i,j,k) - expected) < 1e-12);
        }

    return true;
}



int main(){
    const size_t N = 16384;  // a few thousands
    int failures = 0;

    if (!runTests1D<false>(N)) {
        std::cerr<<"FAILURES in non-unified mode\n";
        ++failures;
    }
    if (!runTests1D<true>(N)) {
        std::cerr<<"FAILURES in unified mode\n";
        ++failures;
    }

    if (failures == 0) {
        std::cout<<"===> ALL TESTS PASSED <===\n";
    } else {
        std::cout<<"===> "<<failures<<" MODE(S) FAILED <===\n";
    }

    int fails=0;
    if (!testMirror2D<false>(64,128)) { std::cerr<<"mirror2D non-unified failed\n"; ++fails; }
    //if (!testMirror2D<true>(64,128))  { std::cerr<<"mirror2D unified failed\n";  ++fails; }

    // Field 3D
    if (!testField3D<false>(32,32,16)) { std::cerr<<"field3D non-unified failed\n"; ++fails; }
    //if (!testField3D<true>(32,32,16))  { std::cerr<<"field3D unified failed\n";  ++fails; }

    if (fails==0) std::cout<<"ALL EXTENDED TESTS PASSED\n";
    else          std::cout<<fails<<" TEST(S) FAILED\n";
    return fails;
    //return failures;
}
