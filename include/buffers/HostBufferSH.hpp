#pragma once
#include "VirtualBuffer.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstring>
#include <stdexcept>
#include <iostream>

//==============================================================================
// HostBuffer: aligned host memory buffer
//==============================================================================

template<typename T, uint Dim, bool unified = false>
class HostBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  HostBuffer(uint n1, uint n2 = 1,uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4)
  {
    this->allocate(n1,n2,n3,n4);
  }
  ~HostBuffer() override{
    this->deallocate();
  };

  // host <-> host
  inline void copyHostHost(T* dstPtr, const T* srcPtr, const uint size){
      std::memcpy(dstPtr, srcPtr, size * sizeof(T));        
  }
  inline void copyFromHost(const T* srcPtr, uint size = 0) {
    size = size > 0 ? size : this->size();
    this->copyHostHost(this->getDataPtr(), srcPtr, size);
  }
  inline void copyToHost(T* dstPtr, uint size = 0) {
    size = size > 0 ? size : this->size();
    this->copyHostHost(dstPtr, this->getDataPtr(), size);
  }

  // host <-> device
  inline void copyHostDevice(T* dstPtr, const T* srcPtr, const uint size, cudaMemcpyKind cudaCopyKind, cudaStream_t stream = 0){      
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, size * sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(dstPtr, srcPtr, size * sizeof(T));        
#endif
    } else { 
      CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, size * sizeof(T),cudaCopyKind,stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }  
  }
  inline void copyFromDevice(const T* srcPtr, const uint size, cudaStream_t stream = 0){
    this->copyHostDevice(this->getDataPtr(), srcPtr, size, cudaMemcpyDeviceToHost, stream);
  }
  inline void copyToDevice(T* dst, const uint size, cudaStream_t stream = 0){
    this->copyHostDevice(dst, this->getDataPtr(), size, cudaMemcpyHostToDevice, stream);
  }

  // self type copy
  inline void selfTypeCopy(T* dstPtr, const T* srcPtr, const uint size) override {
    this->copyHostHost(dstPtr, srcPtr, size);
  }

protected:
  inline void allocate(const uint n1, const uint n2 = 1, const uint n3 = 1, const uint n4 = 1) override {
    if(this->total_ > 0){
      this->data_ = static_cast<T*>(::operator new(this->total_ * sizeof(T),std::align_val_t(64)));
    }
  }

  inline void deallocatePtr(T* ptr) override {
        ::operator delete(ptr, std::align_val_t(4096));
  }
  
};
