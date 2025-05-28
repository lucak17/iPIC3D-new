// test_dataconnector_sharedptr.cpp

#include "DataConnector.hpp"

#include "HostPinnedBufferSH.hpp"
#include "HostBufferSH.hpp"
#include "DeviceBufferSH.hpp"
#include "MirrorHostDeviceBufferSH.hpp"
#include "Field.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>
#include <stdexcept>

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

// 1D fill kernel
__global__ void fillKernel(int* data, size_t n, int factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = int(idx) * factor;
}

// 2D add kernel
__global__ void addKernel2D(float* data, int rows, int cols, float factor) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        size_t idx = size_t(j)*rows + i;
        data[idx] += factor;
    }
}

// 3D mul kernel
__global__ void mulKernel3D(double* data, int X, int Y, int Z, double factor) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i < X && j < Y && k < Z) {
        size_t idx = (size_t)k*X*Y + (size_t)j*X + i;
        data[idx] *= factor;
    }
}

//------------------------------------------------------------------------------
//  Basic DataConnector tests (no buffers):
bool testConnectorBasic() {
    std::cout << "[Basic DataConnector] ";
    auto& dc = DataConnector::instance();
    dc.deregisterAll();

    // 1) register/get int
    dc.registerData("i", 123);
    assert(dc.getDataAs<int>("i") == 123);

    // 2) register/get by-ref vector
    dc.registerData("v", std::vector<int>{1,2});
    auto& vr = dc.getDataAsRef<std::vector<int>>("v");
    vr.push_back(3);
    auto vc = dc.getDataAs<std::vector<int>>("v");
    assert(vc.size()==3 && vc[2]==3);

    // 3) shared_ptr lifetime
    auto sp = std::make_shared<std::string>("hi");
    assert(sp.use_count()==1);
    dc.registerData("s", sp);
    assert(sp.use_count()==2);
    {
        auto sp2 = dc.getDataAs<std::shared_ptr<std::string>>("s");
        assert(*sp2=="hi");
        assert(sp.use_count()==3);
    }
    assert(sp.use_count()==2);

    // 4) duplicate registration throws
    try {
        dc.registerData("i", 999);
        std::cerr<<"FAIL dup\n"; return false;
    } catch(std::runtime_error&) { /*ok*/ }

    // 5) missing key throws
    try {
        dc.getDataAs<int>("nope");
        std::cerr<<"FAIL missing\n"; return false;
    } catch(std::runtime_error&) { /*ok*/ }

    // 6) bad_any_cast throws
    dc.registerData("d", 3.14);
    try {
        dc.getDataAs<int>("d");
        std::cerr<<"FAIL bad_any_cast\n"; return false;
    } catch(std::bad_any_cast&) { /*ok*/ }

    // 7) deregisterData
    dc.deregisterData("i");
    try {
        dc.getDataAs<int>("i");
        std::cerr<<"FAIL deregisterData\n"; return false;
    } catch(std::runtime_error&) { /*ok*/ }

    // 8) deregisterAll
    dc.registerData("a", 1);
    dc.registerData("b", 2);
    dc.deregisterAll();
    try {
        dc.getDataAs<int>("a");
        std::cerr<<"FAIL deregisterAll\n"; return false;
    } catch(std::runtime_error&) { /*ok*/ }

    std::cout<<"OK\n";
    return true;
}

//------------------------------------------------------------------------------
//  1D integration through DataConnector using shared_ptrs:
template<bool U>
bool integration1D_shared(size_t N) {
    std::cout<<"[Integration1D_shared "<<(U?"unified":"non-unified")<<" N="<<N<<"]\n";
    auto& dc = DataConnector::instance();
    dc.deregisterAll();

    // HostPinnedBuffer
    {
        auto hp = std::make_shared<HostPinnedBuffer<int,1,U>>(N);
        dc.registerData("hp", hp);
        auto hpr = dc.getDataAs<std::shared_ptr<HostPinnedBuffer<int,1,U>>>("hp");
        std::vector<int> src(N), dst(N);
        for(size_t i=0;i<N;++i) src[i]=int(i*3+1);
        hpr->copyFromHost(src.data(),N);
        hpr->copyToHost(dst.data(),N);
        for(size_t i=0;i<N;++i)
            if(dst[i]!=src[i]) { std::cerr<<"HP1D mismatch\n"; return false; }
    }

    // HostBuffer
    {
        auto hb = std::make_shared<HostBuffer<int,1,false>>(N);
        dc.registerData("hb", hb);
        auto hbr = dc.getDataAs<std::shared_ptr<HostBuffer<int,1,false>>>("hb");
        for(size_t i=0;i<N;++i) (*hbr)(i)=int(i*2);
        for(size_t i=0;i<N;++i)
            if((*hbr)(i)!=int(i*2)){ std::cerr<<"HB1D mismatch\n"; return false; }
    }

    // HostPinnedBuffer <-> DeviceBuffer
    {
        auto d   = std::make_shared<DeviceBuffer<int,1,U>>(N);
        auto hp2 = std::make_shared<HostPinnedBuffer<int,1,U>>(N);
        dc.registerData("d",  d);
        dc.registerData("hp2",hp2);

        for(size_t i=0;i<N;++i) hp2->getDataPtr()[i]=int(i+5);
        hp2->copyToDevice(d->getDataPtr(),N); CUDA_SYNC();

        auto hp3 = std::make_shared<HostPinnedBuffer<int,1,U>>(N);
        dc.registerData("hp3",hp3);
        hp3->copyFromDevice(d->getDataPtr(),N); CUDA_SYNC();
        for(size_t i=0;i<N;++i)
            if(hp3->getDataPtr()[i]!=hp2->getDataPtr()[i]){
                std::cerr<<"HP<->D mismatch\n"; return false;
            }
    }

    // DeviceBuffer kernel
    {
        auto d2 = std::make_shared<DeviceBuffer<int,1,U>>(N);
        dc.registerData("d2", d2);
        int *ptr = d2->getDataPtr();
        constexpr int F = 7;
        dim3 b(256), g((N+b.x-1)/b.x);
        fillKernel<<<g,b>>>(ptr,N,F);
        CUDA_SYNC();

        std::vector<int> host(N);
        d2->copyToHost(host.data(),N); CUDA_SYNC();
        for(size_t i=0;i<N;++i)
            if(host[i]!=int(i)*F){ std::cerr<<"D kernel mismatch\n"; return false; }
    }

    // MirrorHostDeviceBuffer & Field 1D
    {
        auto m = std::make_shared<MirrorHostDeviceBuffer<int,1,U>>(N);
        dc.registerData("m", m);
        int *hptr = m->getHostDataPtr();
        constexpr int F1 = 5;
        m->copyHostToDevice(); CUDA_SYNC();
        {
            dim3 b(256), g((N+b.x-1)/b.x);
            fillKernel<<<g,b>>>(m->getDeviceDataPtr(),N,F1);
        }
        CUDA_SYNC();
        m->copyDeviceToHost(); CUDA_SYNC();
        for(size_t i=0;i<N;++i)
            if(hptr[i]!=int(i)*F1){ std::cerr<<"Mirror1D mismatch\n"; return false; }
    }
    {
        auto f = std::make_shared<Field<int,1,U>>(N);
        dc.registerData("f", f);
        int *hptr = f->getHostDataPtr();
        constexpr int F = 11;
        f->copyHostToDevice(); CUDA_SYNC();
        {
            dim3 b(256), g((N+b.x-1)/b.x);
            fillKernel<<<g,b>>>(f->getDeviceDataPtr(),N,F);
        }
        CUDA_SYNC();
        f->copyDeviceToHost(); CUDA_SYNC();
        for(size_t i=0;i<N;++i)
            if(hptr[i]!=int(i)*F){ std::cerr<<"Field1D mismatch\n"; return false; }
    }

    // Unified alias test
    if constexpr(U) {
        auto mu = std::make_shared<MirrorHostDeviceBuffer<int,1,true>>(N);
        dc.registerData("mu", mu);
        if(mu->getHostDataPtr()!=mu->getDeviceDataPtr()){
            std::cerr<<"Unified alias fail\n"; return false;
        }
    }

    dc.deregisterAll();
    std::cout<<"[1D_shared OK]\n";
    return true;
}

//------------------------------------------------------------------------------
//  2D integration (Mirror only):
template<bool U>
bool integrationMirror2D_shared(int rows,int cols) {
    std::cout<<"[Integration Mirror2D_shared "<<U<<" "<<rows<<"x"<<cols<<"]\n";
    auto& dc = DataConnector::instance();
    dc.deregisterAll();

    int N = rows*cols;
    auto m = std::make_shared<MirrorHostDeviceBuffer<float,2,U>>(rows,cols);
    dc.registerData("m2d", m);
    auto mr = dc.getDataAs<std::shared_ptr<MirrorHostDeviceBuffer<float,2,U>>>("m2d");

    float* h = mr->getHostDataPtr();
    for(int idx=0; idx<N; ++idx) h[idx] = float(idx%rows) - float(idx/rows);
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j){
        assert(mr->operator()(i,j) == h[j*rows + i]);
    }

    mr->copyHostToDevice(); CUDA_SYNC();
    std::fill(h,h+N,0);
    mr->copyDeviceToHost(); CUDA_SYNC();
    for(int idx=0;idx<N;++idx){
        float exp = float(idx%rows)-float(idx/rows);
        assert(fabs(h[idx]-exp) < 1e-5f);
    }

    constexpr float F = 2.5f;
    dim3 b(16,16), g((rows+15)/16,(cols+15)/16);
    addKernel2D<<<g,b>>>(mr->getDeviceDataPtr(),rows,cols,F);
    CUDA_SYNC();
    mr->copyDeviceToHost(); CUDA_SYNC();
    for(int idx=0;idx<N;++idx){
        float exp = (float(idx%rows)-float(idx/rows)) + F;
        assert(fabs(h[idx]-exp) < 1e-5f);
    }

    dc.deregisterAll();
    std::cout<<"[Mirror2D_shared OK]\n";
    return true;
}

//------------------------------------------------------------------------------
//  3D integration (Field only):
template<bool U>
bool integrationField3D_shared(int X,int Y,int Z) {
    std::cout<<"[Integration Field3D_shared "<<U<<" "<<X<<"x"<<Y<<"x"<<Z<<"]\n";
    auto& dc = DataConnector::instance();
    dc.deregisterAll();

    size_t N = size_t(X)*Y*Z;
    auto f = std::make_shared<Field<double,3,U>>(X,Y,Z);
    dc.registerData("f3d", f);
    auto fr = dc.getDataAs<std::shared_ptr<Field<double,3,U>>>("f3d");

    double* h = fr->getHostDataPtr();
    for(int i=0;i<X;++i) for(int j=0;j<Y;++j) for(int k=0;k<Z;++k)
        fr->operator()(i,j,k) = i + j*10 + k*100;

    for(int i=0;i<X;++i) for(int j=0;j<Y;++j) for(int k=0;k<Z;++k){
        size_t idx = size_t(k)*X*Y + size_t(j)*X + i;
        assert(h[idx] == fr->operator()(i,j,k));
    }

    fr->copyHostToDevice(); CUDA_SYNC();
    std::fill(h,h+N,0);
    fr->copyDeviceToHost(); CUDA_SYNC();
    for(int i=0;i<X;++i) for(int j=0;j<Y;++j) for(int k=0;k<Z;++k)
        assert(fr->operator()(i,j,k) == i + j*10 + k*100);

    constexpr double F = 1.5;
    dim3 b(8,8,4), g((X+7)/8,(Y+7)/8,(Z+3)/4);
    mulKernel3D<<<g,b>>>(fr->getDeviceDataPtr(),X,Y,Z,F);
    CUDA_SYNC();
    fr->copyDeviceToHost(); CUDA_SYNC();
    for(int i=0;i<X;++i) for(int j=0;j<Y;++j) for(int k=0;k<Z;++k){
        double exp = (i + j*10 + k*100)*F;
        assert(fabs(fr->operator()(i,j,k)-exp) < 1e-12);
    }

    dc.deregisterAll();
    std::cout<<"[Field3D_shared OK]\n";
    return true;
}

//------------------------------------------------------------------------------
int main(){
    int fails = 0;

    fails += !testConnectorBasic();
    fails += !integration1D_shared<false>(16384);
    fails += !integration1D_shared<true>(16384);
    fails += !integrationMirror2D_shared<false>(64,128);
    // fails += !integrationMirror2D_shared<true>(64,128);
    fails += !integrationField3D_shared<false>(32,32,16);
    // fails += !integrationField3D_shared<true>(32,32,16);

    if (fails==0) std::cout<<"===> ALL SHARED_PTR INTEGRATION TESTS PASSED <===\n";
    else           std::cout<<"===> "<<fails<<" TEST(S) FAILED <===\n";

    return fails;
}