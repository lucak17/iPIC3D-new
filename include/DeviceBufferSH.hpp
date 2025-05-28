#pragma once
#include "VirtualBuffer.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstring>
#include <stdexcept>


//==============================================================================
// DeviceBuffer: GPU global or unified‐aligned memory
//==============================================================================

template<typename T, uint Dim, bool unified = false>
class DeviceBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  DeviceBuffer(uint n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4)
  {
    this->allocate(n1,n2,n3,n4);
  }
  ~DeviceBuffer() override
  {
    this->deallocate();
  }

  // device <-> device
  inline void copyDeviceDevice(T* dstPtr, const T* srcPtr, const uint size){
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, size * sizeof(T), cudaMemcpyDefault); );
#else
      std::memcpy(dstPtr, srcPtr, size *  sizeof(T));
#endif
    } else {
      CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, size * sizeof(T),cudaMemcpyDeviceToDevice));
    }
  }
  inline void copyFromDevice(const T* srcPtr, uint size = 0) {
    size = size > 0 ? size : this->size();
    this->copyDeviceDevice(this->getDataPtr(), srcPtr, size);
  }
  inline void copyToDevice(T* dstPtr, uint size = 0) {
    size = size > 0 ? size : this->size();
    this->copyDeviceDevice(dstPtr, this->getDataPtr(), size);
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
      CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, size * sizeof(T), cudaCopyKind, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  inline void copyFromHost(const T* srcPtr, const uint size, cudaStream_t stream = 0){
    this->copyHostDevice(this->getDataPtr(), srcPtr, size, cudaMemcpyHostToDevice, stream);
  }
  inline void copyToHost(T* dst, const uint size, cudaStream_t stream = 0) {
    this->copyHostDevice(dst, this->getDataPtr(), size, cudaMemcpyDeviceToHost, stream);
  }

  
  // self type copy
  inline void selfTypeCopy(T* dstPtr, const T* srcPtr, const uint size) override {
    this->copyDeviceDevice(dstPtr, srcPtr, size);
  }

  inline void allocateForUnifiedMemory(T* ptr){
    if(this->data_ != nullptr){
        this->deallocatePtr(this->data_);
    }
    this->isMirroringAnotherBuffer = true;
    this->setPtr(ptr);
  }

protected:
  // allocation & deallocation
  inline void allocate(const uint n1, const uint n2 = 1, const uint n3 = 1, const uint n4 = 1) override {
    if(this->total_ > 0){
        if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMallocManaged(&this->data_, this->total_ * sizeof(T)) );
#else
        this->data_ = static_cast<T*>(::operator new(this->total_ * sizeof(T),std::align_val_t(4096)));
#endif
    } else { CUDA_CHECK(cudaMalloc(&this->data_, this->total_ * sizeof(T))); }
    }
  }

  inline void deallocatePtr(T* ptr) override {
    if constexpr (unified) {
#if CUDA_MANAGED
            cudaFree(ptr);
#else
        ::operator delete(ptr, std::align_val_t(4096));
#endif
        } else{ cudaFree(ptr);}
  }
};
