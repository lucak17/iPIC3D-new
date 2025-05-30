#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <cstdint>

using uint = std::uint32_t;

#define CUDA_MANAGED 1

// Abstract base for 1D–4D flat storage + indexing
template<typename T, uint Dim, bool unified>
class VirtualBuffer {
  static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4");

public:
  VirtualBuffer(uint n1, uint n2 = 1, uint n3 = 1, uint n4 = 1)
    : n1_(n1), n2_(n2), n3_(n3), n4_(n4), total_(n1_*n2_*n3_*n4_), data_(nullptr)
  {
    this->recalculateStrides();
  }

  virtual ~VirtualBuffer() {}

  virtual void selfTypeCopy(T* dstPtr, const T* srcPtr, const uint size) = 0;

  __host__ __device__ __forceinline__ void setExtents(const uint n1, const uint n2 = 1, const uint n3 = 1, const uint n4 = 1){
    this->n1_ = n1;
    this->n2_ = n2;
    this->n3_ = n3;
    this->n4_ = n4;
    this->total_ = n1*n2*n3*n4;
    this->recalculateStrides();
  }

  virtual void expandBuffer(uint n1, uint n2=1, uint n3=1, uint n4=1, cudaStream_t stream = 0 ){
    T* oldPtr = this->data_;
    uint oldSize = this->size();
    this->setExtents(n1,n2,n3,n4);
    this->allocate(n1,n2,n3,n4);
    this->selfTypeCopy(this->data_, oldPtr, oldSize);
    this->deallocatePtr(oldPtr);
  }

  void setPtr(T* ptr){
    this->data_ = ptr;
  }
  
  __host__ __device__ __forceinline__ T*       getDataPtr()       noexcept { return data_; }
  __host__ __device__ __forceinline__ const T* getDataPtr() const noexcept { return data_; }

  __host__ __device__ __forceinline__ uint size() const noexcept { return total_; }

  // 1D…4D operator()
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
    return data_[i + j*stride_j_];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==2, const T&>
  operator()(const uint i, const uint j) const noexcept {
    return data_[i + j*stride_j_];
  }

  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==3, T&>
  operator()(const uint i, const uint j, const uint k) noexcept {
    return data_[i + j*stride_j_ + k*stride_k_];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==3, const T&>
  operator()(const uint i, const uint j, const uint k) const noexcept {
    return data_[i + j*stride_j_ + k*stride_k_];
  }

  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==4, T&>
  operator()(const uint i, const uint j, const uint k, const uint l) noexcept {
    return data_[i + j*stride_j_ + k*stride_k_ + l*stride_l_];
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==4, const T&>
  operator()(const uint i, const uint j, const uint k, const uint l) const noexcept {
    return data_[i + j*stride_j_ + k*stride_k_ + l*stride_l_];
  }

  // get 1D flat index
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==1, uint>
  get1DFlatIndex(const uint i) const noexcept {
    return i;
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==2, uint>
  get1DFlatIndex(const uint i, const uint j) const noexcept {
    return i + j*stride_j_;
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==3, uint>
  get1DFlatIndex(const uint i, const uint j, const uint k) const noexcept {
    return i + j*stride_j_ + k*stride_k_;
  }
  template<uint D = Dim>
  __host__ __device__ __forceinline__ std::enable_if_t<D==4, uint>
  get1DFlatIndex(const uint i, const uint j, const uint k, const uint l) const noexcept {
    return i + j*stride_j_ + k*stride_k_ + l*stride_l_;
  }

protected:
  virtual void allocate(const uint n1, const uint n2 = 1, const uint n3 = 1, const uint n4 = 1) = 0;
  virtual void deallocatePtr(T* ptr) = 0;
  inline virtual void deallocate(){
    if(this->data_ != nullptr && !this->isMirroringAnotherBuffer){
      this->deallocatePtr(this->data_);
    }
  }
  __host__ __device__ __forceinline__ void recalculateStrides() {
    stride_i_ = 1;
    stride_j_ = n1_;
    stride_k_ = n1_ * n2_;
    stride_l_ = n1_ * n2_ * n3_;
  }
  bool isMirroringAnotherBuffer = false;
  uint n1_, n2_, n3_, n4_, total_;
  uint stride_i_, stride_j_, stride_k_, stride_l_;
  T*           data_ = nullptr;
};
