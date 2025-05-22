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

#define CUDA_MANAGED 1

// Abstract base for 1D–4D flat storage + indexing
template<typename T, uint Dim, bool unified>
class VirtualBuffer {
  static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4");

public:
  VirtualBuffer(uint n1,
                uint n2 = 1,
                uint n3 = 1,
                uint n4 = 1)
    : n1_(n1), n2_(n2), n3_(n3), n4_(n4),
      total_(n1_*n2_*n3_*n4_), data_(nullptr)
  {
  }

  virtual ~VirtualBuffer() {
  }

  /*
  virtual void copyFrom(const VirtualBuffer<T,Dim,unified>& src) = 0;
  virtual void copyTo(VirtualBuffer<T,Dim,unified>& dst) const = 0;
  virtual void copyFrom(const VirtualBuffer<T,Dim,unified>& src, cudaStream_t stream = 0) = 0;
  virtual void copyTo(VirtualBuffer<T,Dim,unified>& dst, cudaStream_t stream = 0) const = 0;
  */

  void setPtr(T* ptr){
    this->data_ = ptr;
  }
  
  T*       getDataPtr()       noexcept { return data_; }
  const T* getDataPtr() const noexcept { return data_; }

  uint size() const noexcept { return total_; }

  // 1D…4D operator() as before…
  template<uint D = Dim>
  inline std::enable_if_t<D==1, T&>
  operator()(uint i) noexcept {
    return data_[i];
  }
  template<uint D = Dim>
  inline std::enable_if_t<D==1, const T&>
  operator()(uint i) const noexcept {
    return data_[i];
  }

  template<uint D = Dim>
  inline std::enable_if_t<D==2, T&>
  operator()(uint i, uint j) noexcept {
    return data_[i*n2_ + j];
  }
  template<uint D = Dim>
  inline std::enable_if_t<D==2, const T&>
  operator()(uint i, uint j) const noexcept {
    return data_[i*n2_ + j];
  }

  template<uint D = Dim>
  inline std::enable_if_t<D==3, T&>
  operator()(uint i, uint j, uint k) noexcept {
    return data_[(i*n2_ + j)*n3_ + k];
  }
  template<uint D = Dim>
  inline std::enable_if_t<D==3, const T&>
  operator()(uint i, uint j, uint k) const noexcept {
    return data_[(i*n2_ + j)*n3_ + k];
  }

  template<uint D = Dim>
  inline std::enable_if_t<D==4, T&>
  operator()(uint i, uint j,
             uint k, uint l) noexcept {
    return data_[((i*n2_+j)*n3_+k)*n4_ + l];
  }
  template<uint D = Dim>
  inline std::enable_if_t<D==4, const T&>
  operator()(uint i, uint j,
             uint k, uint l) const noexcept {
    return data_[((i*n2_+j)*n3_+k)*n4_ + l];
  }

protected:
  virtual void allocate(int n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) = 0;
  virtual void deallocate() = 0;
  bool isMirroringAnotherBuffer = false;
  uint n1_, n2_, n3_, n4_, total_;
  T*           data_ = nullptr;
};
