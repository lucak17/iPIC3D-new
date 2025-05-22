#pragma once
#include "virtualBuffer.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstring>
#include <stdexcept>
#include <iostream>

// Helper macro for CUDA error checking
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess)                                             \
      throw std::runtime_error(cudaGetErrorString(err));                \
  } while (0)
#endif

// Forward‐declare DeviceBuffer so we can refer to it
template<typename T, uint Dim, bool unified>
class DeviceBuffer;

//==============================================================================
// HostBuffer: pinned or unified‐aligned host memory
//==============================================================================

template<typename T, uint Dim, bool unified = false>
class HostBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  // ctor/dtor
  HostBuffer(uint n1, uint n2 = 1,uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4)
  {
    this->allocate(n1,n2,n3,n4);
  }
  ~HostBuffer() override{
    this->deallocate();
  };

  // host <-> host
  inline void copyFrom(const HostBuffer& src) {
    if (&src == this) return;
    if (src.size() != this->size())
      throw std::runtime_error("HostBuffer::copyFrom size mismatch");
    std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T));        
  }

  inline void copyTo(HostBuffer& dst) const {
    dst.copyFrom(*this);
  }

  // host <-> host
  inline void copyFrom(const HostPinnedBuffer& src) {
    if (&src == this) return;
    if (src.size() != this->size())
      throw std::runtime_error("HostBuffer::copyFrom size mismatch");
    std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T));        
  }

  inline void copyTo(HostPinnedBuffer& dst) const {
    if (&dst == this) return;
    if (dst.size() != this->size())
      throw std::runtime_error("HostBuffer::copyFrom size mismatch");
    std::memcpy(dst.getDataPtr(),this->getDataPtr(), this->size()*sizeof(T));
  }

  inline void selfCopy(T* newPtr, T* oldPtr, uint oldSize){
      std::memcpy(newPtr, oldPtr, oldSize * sizeof(T));        
  }

  // device -> host
  inline void copyFrom(const DeviceBuffer<T,Dim,unified>& src, cudaStream_t stream = 0)
  {
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T));        
#endif
    } else { 
      CUDA_CHECK(cudaMemcpyAsync(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  // host -> device
  inline void copyTo(DeviceBuffer<T,Dim,unified>& dst, cudaStream_t stream = 0) const
  {
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(dst.getDataPtr(), this->getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(dst.getDataPtr(), this->getDataPtr(), this->size()*sizeof(T));
#endif
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dst.getDataPtr(), this->getDataPtr(),this->size()*sizeof(T),cudaMemcpyHostToDevice,stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

protected:
  inline void allocate(int n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) override {
    if(this->total_ > 0){
      this->data_ = static_cast<T*>(::operator new(this->total_ * sizeof(T),std::align_val_t(4096)));
    }
  }

  inline void deallocatePtr(T* ptr) override {
        ::operator delete(ptr, std::align_val_t(4096));
  }
  
};
