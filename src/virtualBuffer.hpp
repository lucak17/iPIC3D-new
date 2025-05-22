#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>


#define CUDA_CHECK(call)                                  \
  do {                                                    \
    cudaError_t err = call;                               \
    if (err != cudaSuccess)                               \
      throw std::runtime_error(cudaGetErrorString(err));  \
  } while (0)

#define CUDA_MANAGED 0

// Abstract base for 1D–4D flat storage + indexing
template<typename T, uint Dim, bool unified>
class VirtualBuffer {
  static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4");

public:
  VirtualBuffer(uint n1, uint n2 = 1, uint n3 = 1, uint n4 = 1)
    : n1_(n1), n2_(n2), n3_(n3), n4_(n4), total_(n1_*n2_*n3_*n4_),
    stride4_(1), stride3_(n4), stride2_(n3_*stride3_), stride1_(n2_*stride2_), data_(nullptr)
  {
  }

  virtual ~VirtualBuffer() {}

  virtual void selfCopy(T* newPtr, T* oldPtr, uint oldSize) = 0;

  __host__ __device__ __forceinline__ void setExtents(const uint n1, const uint n2 = 1, const uint n3 = 1, const uint n4 = 1){
    this->n1_ = n1;
    this->n2_ = n2;
    this->n3_ = n3;
    this->n4_ = n4;
    this->total_ = n1*n2*n3*n4;
    this->stride4_ = 1;
    this->stride3_ = n4;
    this->stride2_ = n3_*this->stride3_;
    this->stride1_ = n2_*this->stride2_;
  }

  virtual void expandBuffer(uint n1, uint n2=1, uint n3=1, uint n4=1, cudaStream_t stream = 0 ){
    auto oldPtr = this->data_;
    auto oldSize = this->size();
    this->setExtents(n1,n2,n3,n4);
    this->allocate(n1,n2,n3,n4);
    this->selfCopy(this->data_,oldPtr, oldSize);
    this->deallocatePtr(oldPtr);
  }

  void setPtr(T* ptr){
    this->data_ = ptr;
  }
  
  __host__ __device__ __forceinline__ T*       getDataPtr()       noexcept { return data_; }
  __host__ __device__ __forceinline__ const T* getDataPtr() const noexcept { return data_; }

  __host__ __device__ __forceinline__ uint size() const noexcept { return total_; }

  // 1D…4D operator() as before…
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==1, T&>
  operator()(const uint i) noexcept {
    return data_[i];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==1, const T&>
  operator()(const uint i) const noexcept {
    return data_[i];
  }

  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==2, T&>
  operator()(const uint i, const uint j) noexcept {
    return data_[i*stride1_+ j];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==2, const T&>
  operator()(const uint i, const uint j) const noexcept {
    return data_[i*stride1_ + j];
  }

  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==3, T&>
  operator()(const uint i, const uint j, const uint k) noexcept {
    return data_[i*stride1_ + j*stride2_ + k];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==3, const T&>
  operator()(const uint i, const uint j, const uint k) const noexcept {
    return data_[i*stride1_ + j*stride2_ + k];
  }

  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==4, T&>
  operator()(const uint i, const uint j, const uint k, const uint l) noexcept {
    return data_[i*stride1_ + j*stride2_ + k*stride3_ + l];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==4, const T&>
  operator()(const uint i, const uint j, const uint k, const uint l) const noexcept {
    return data_[i*stride1_ + j*stride2_ + k*stride3_ + l];
  }

protected:
  virtual void allocate(int n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) = 0;
  virtual void deallocatePtr(T* ptr) = 0;
  inline virtual void deallocate(){
    if(this->data_ != nullptr && !this->isMirroringAnotherBuffer){
      this->deallocatePtr(this->data_);
    }
  }
  bool isMirroringAnotherBuffer = false;
  uint n1_, n2_, n3_, n4_, total_;
  uint stride4_, stride3_, stride2_, stride1_;
  T*           data_ = nullptr;
};
