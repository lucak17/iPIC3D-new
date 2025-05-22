#pragma once
#include "virtualBuffer.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstring>
#include <stdexcept>



// forward‐declare HostBuffer so we can refer to it
template<typename T, uint Dim, bool unified>
class HostBuffer;

//==============================================================================
// DeviceBuffer: GPU global or unified‐aligned memory
//==============================================================================

template<typename T, uint Dim, bool unified = false>
class DeviceBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  // ctor / dtor
  DeviceBuffer(uint n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4)
  {
    this->allocate(n1,n2,n3,n4);
  }
  ~DeviceBuffer() override
  {
    this->deallocate();
  }

  // device <-> device
  inline void copyFrom(const DeviceBuffer& src) {
    if (&src == this) return;
    if (src.size() != this->size())
      throw std::runtime_error("DeviceBuffer::copyFrom size mismatch");
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
      std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size() * sizeof(T));
#endif
    } else {
      CUDA_CHECK(cudaMemcpy(this->getDataPtr(), src.getDataPtr(),this->size() * sizeof(T),cudaMemcpyDeviceToDevice));
    }
  }

  inline void copyTo(DeviceBuffer& dst) const {
    dst.copyFrom(*this);
  }

  // host -> device
  inline void copyFrom(const HostBuffer<T,Dim,unified>& src,cudaStream_t stream = 0)
  {
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(this->getDataPtr(), src.getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
      std::memcpy(this->getDataPtr(), src.getDataPtr(), this->size() * sizeof(T));
#endif
    } else {
      CUDA_CHECK(cudaMemcpyAsync(this->getDataPtr(), src.getDataPtr(),this->size() * sizeof(T),cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  // device -> host
  inline void copyTo(HostBuffer<T,Dim,unified>& dst, cudaStream_t stream = 0) const
  {
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(dst.getDataPtr(), this->getDataPtr(), this->size()*sizeof(T), cudaMemcpyDefault); );
#else
      std::memcpy(dst.getDataPtr(), this->getDataPtr(), this->size() * sizeof(T));
#endif
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dst.getDataPtr(), this->getDataPtr(),this->size() * sizeof(T),cudaMemcpyDeviceToHost,stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  inline void selfCopy(T* newPtr, T* oldPtr, uint oldSize){
    if constexpr (unified) {
#if CUDA_MANAGED
        CUDA_CHECK(cudaMemcpy(newPtr, oldPtr, oldSize * sizeof(T), cudaMemcpyDefault); );
#else
        std::memcpy(newPtr, oldPtr, oldSize * sizeof(T));        
#endif
    } else {
      CUDA_CHECK(cudaMemcpy(newPtr, oldPtr, oldSize * sizeof(T),cudaMemcpyDeviceToDevice));
    }
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
  inline void allocate(int n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) override {
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
