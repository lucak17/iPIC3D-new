// test.cpp

#include "hostBufferSH.hpp"
#include "deviceBufferSH.hpp"
#include "mirrorHostDeviceBufferSH.hpp"
#include "field.hpp"

#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// Helper to check CUDA errors
#define CUDA_SYNC()                                    \
  do {                                                 \
    cudaError_t e = cudaDeviceSynchronize();           \
    if (e != cudaSuccess) {                            \
      std::cerr << "CUDA sync error: "                 \
                << cudaGetErrorString(e) << "\n";      \
      return false;                                    \
    }                                                   \
  } while (0)

// Alignment check
bool is_aligned(void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) % a) == 0;
}

// 1) Test HostBuffer non-unified
bool testHostBufferNonUnified() {
  {
    HostBuffer<int,1,false> h(10);
    assert(h.size()==10);
    for (int i=0;i<10;++i) h(i)=i*3;
    for (int i=0;i<10;++i) assert(h(i)==i*3);
  }
  {
    HostBuffer<float,2,false> h(4,5);
    assert(h.size()==20);
    for (unsigned i=0;i<4;++i)
      for (unsigned j=0;j<5;++j)
        h(i,j) = float(i+10*j);
    for (unsigned i=0;i<4;++i)
      for (unsigned j=0;j<5;++j)
        assert(std::abs(h(i,j) - float(i+10*j)) < 1e-6f);
  }
  {
    HostBuffer<double,3,false> h(2,3,4);
    assert(h.size()==24);
    for (unsigned i=0;i<2;++i)
     for (unsigned j=0;j<3;++j)
      for (unsigned k=0;k<4;++k)
        h(i,j,k) = double(i + 10*j + 100*k);
    for (unsigned i=0;i<2;++i)
     for (unsigned j=0;j<3;++j)
      for (unsigned k=0;k<4;++k)
        assert(std::abs(h(i,j,k) - double(i + 10*j + 100*k)) < 1e-12);
  }
  {
    uint n1=23, n2=42, n3=32, n4=4;
    HostBuffer<int,4,false> h(n1,n2,n3,n4);
    assert(h.size()==n1*n2*n3*n4);
    for(unsigned i=0;i<n1;++i)
     for(unsigned j=0;j<n2;++j)
      for(unsigned k=0;k<n3;++k)
       for(unsigned l=0;l<n4;++l)
         h(i,j,k,l) = int(i + 2*j + 4*k + 8*l);
    for(unsigned i=0;i<n1;++i)
     for(unsigned j=0;j<n2;++j)
      for(unsigned k=0;k<n3;++k)
       for(unsigned l=0;l<n4;++l)
         assert(h(i,j,k,l) == int(i + 2*j + 4*k + 8*l));
    
    std::cout<< "h(24,2,56,4)" << h(2,2,56,3) <<std::endl;
  }
  return true;
}

// 2) Test DeviceBuffer non-unified H↔D
bool testDeviceBufferNonUnified() {
  constexpr unsigned N = 5000000;
  HostBuffer<int,1,false>  h(N);
  DeviceBuffer<int,1,false> d(N);

  for(unsigned i=0;i<N;++i) h(i)=int(i*5);
  d.copyFrom(h);        // host→device
  CUDA_SYNC();

  HostBuffer<int,1,false> back(N);
  d.copyTo(back);       // device→host
  CUDA_SYNC();

  for(unsigned i=0;i<N;++i) assert(back(i)==h(i));
  return true;
}

// 3) Test MirrorHostDeviceBuffer non-unified
bool testMirrorNonUnified() {
  MirrorHostDeviceBuffer<float,2,false> m(6,7);
  auto *h = m.getHostBufferPtr();
  auto *d = m.getDeviceBufferPtr();
  assert(h && d);

  // fill host
  for(unsigned i=0;i<6;++i)
    for(unsigned j=0;j<7;++j)
      (*h)(i,j) = float(i + 10*j);

  // copy to device, then corrupt host, then copy back
  m.copyHostToDevice();
  CUDA_SYNC();
  for(unsigned i=0;i<6;++i) for(unsigned j=0;j<7;++j) (*h)(i,j)=0.f;
  m.copyDeviceToHost();
  CUDA_SYNC();
  for(unsigned i=0;i<6;++i)
    for(unsigned j=0;j<7;++j)
      assert(std::abs((*h)(i,j) - float(i+10*j)) < 1e-6f);

  return true;
}

// 4) Test Field non-unified
bool testFieldNonUnified() {
  Field<int,3,false> f(4,4,4);
  assert(f.size()==4*4*4);
  // operator() on host
  f(1,2,3) = 321;
  f.copyHostToDevice();
  CUDA_SYNC();
  // zero host memory
  *f.getHostDataPtr() = 0;
  f.copyDeviceToHost();
  CUDA_SYNC();
  assert(f(1,2,3)==321);
  return true;
}

// 5) Test unified mode
bool testUnifiedModes() {
  // HostBuffer unified
  HostBuffer<int,1,true> hu(128);
  assert(is_aligned(hu.getDataPtr(), 4096));
  for(int i=0;i<128;++i) hu(i)=i;
  // DeviceBuffer unified shares same pointer
  DeviceBuffer<int,1,true> du(128);
  assert(hu.getDataPtr()!=nullptr && du.getDataPtr()!=nullptr);
  // Unified: host→device no-op
  hu.copyTo(du);  // just memcpy
  // Unified: device→host no-op
  du.copyTo(hu);
  for(int i=0;i<128;++i) assert(hu(i)==i);

  // Mirror unified
  MirrorHostDeviceBuffer<int,2,true> mu(5,5);
  auto *hh = mu.getHostBufferPtr();
  auto *dd = mu.getDeviceBufferPtr();
  (void)dd; // in unified mode devicePtr==hostPtr
  hh->operator()(3,4)=99;
  mu.copyHostToDevice();
  mu.copyDeviceToHost();
  assert(hh->operator()(3,4)==99);
  std::cout<< "test managed before field" << std::endl;
  // Field unified
  {
    int n1=123, n2=23, n3=32;
    Field<float,3,true> fu(n1,n2,n3);
    std::cout<< "test managed in field 1" << std::endl;
    //assert(fu.getDeviceBufferPtr()==fu.getHostBufferPtr());
    assert(fu.getHostDataPtr()==fu.getDeviceDataPtr());
    for(uint i =0; i < n1; i++ )
      for(uint j =0; j < n2; j++ )
        for(uint k =0; k< n3; k++ ){
          fu(i,j,k) = (i*n2 + j)*n3 + k;
        }
    fu(0,0,0)=3.14f;
    std::cout<< "test managed in field" << std::endl;
    fu.copyHostToDevice();
    fu.copyDeviceToHost();
    assert(std::abs(fu(0,0,0)-3.14f)<1e-6f);
    assert(std::abs(fu(0,0,2)-fu.getDeviceDataPtr()[2])<1e-6f);
  }
  return true;
}

int main(){
  struct { const char* name; bool(*fn)(); } tests[] = {
    {"Host non-unified",      testHostBufferNonUnified},
    {"Device non-unified",    testDeviceBufferNonUnified},
    {"Mirror non-unified",    testMirrorNonUnified},
    {"Field non-unified",     testFieldNonUnified},
    {"Unified modes",         testUnifiedModes},
  };

  int fails = 0;
  for(auto &t : tests){
    std::cout<<"Running "<<t.name<<" ... ";
    if(t.fn()){
      std::cout<<"PASSED\n";
    } else {
      std::cout<<"FAILED\n";
      ++fails;
    }
  }
  if(fails==0) std::cout<<"All tests passed.\n";
  else         std::cout<<fails<<" test(s) failed!\n";
  return fails;
}
