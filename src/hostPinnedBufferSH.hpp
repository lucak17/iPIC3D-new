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
class HostPinnedBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  // ctor/dtor
  HostPinnedBuffer(uint n1, uint n2 = 1,uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4)
  {
    this->allocate(this->total_);
  }
  ~HostPinnedBuffer() override{
    this->deallocate();
  };

  // host <-> host
  inline void copyFrom(const HostPinnedBuffer& src) {
    if (&src == this) return;
    if (src.size() != this->size())
      throw std::runtime_error("HostBuffer::copyFrom size mismatch");
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T));        
#endif
  }

  inline void copyTo(HostPinnedBuffer& dst) const {
    dst.copyFrom(*this);
  }

  inline void selfCopy(T* newPtr, T* oldPtr, uint oldSize){
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(newPtr, oldPtr, oldSize * sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(newPtr, oldPtr, oldSize * sizeof(T));        
#endif
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
      if constexpr (unified) {
#if CUDA_MANAGED
        std::cout<< "Allocate host buffer cuda managed"<<std::endl;
        CUDA_CHECK(cudaMallocManaged(&this->data_, this->total_ * sizeof(T)) );
#else
        this->data_ = static_cast<T*>(::operator new(this->total_ * sizeof(T),std::align_val_t(4096)));
#endif
      } else {CUDA_CHECK(cudaHostAlloc(&this->data_,this->total_*sizeof(T),cudaHostAllocDefault));}
    }
  }

  inline void deallocatePtr(T* ptr) override {
    if constexpr (unified) {
#if CUDA_MANAGED
        cudaFree(ptr);
#else
        ::operator delete(ptr, std::align_val_t(4096));
#endif
      } else {cudaFreeHost(ptr);}
  }  
  
};
