#ifndef ALIGNED_ARRAY_NESTED_HPP
#define ALIGNED_ARRAY_NESTED_HPP

#include <cstddef>
#include <new>
#include <utility>

#ifndef AA_ALIGNMENT
#define AA_ALIGNMENT 64
#endif

#if defined(__INTEL_COMPILER)
  #define AA_ASSUME_ALIGNED_ONE_TIME(P) (__assume_aligned((P), AA_ALIGNMENT), (P))
#elif defined(__GNUC__) || defined(__clang__)
  #define AA_ASSUME_ALIGNED_ONE_TIME(P) \
    reinterpret_cast<decltype(P)>(__builtin_assume_aligned((P), AA_ALIGNMENT))
#else
  #define AA_ASSUME_ALIGNED_ONE_TIME(P) (P)
#endif

//------------------------------------------------------------------------------
// AlignedArray: flat storage, 1–4D, supports [][][]... indexing via proxies
//------------------------------------------------------------------------------
template<typename T>
class AlignedArray {
public:
  // 1D–4D constructors
  explicit AlignedArray(std::size_t n1)
    : dims_{n1,1,1,1}, total_(n1) {
    alloc_data_();
  }
  AlignedArray(std::size_t n1, std::size_t n2)
    : dims_{n1,n2,1,1}, total_(n1*n2) {
    alloc_data_();
  }
  AlignedArray(std::size_t n1, std::size_t n2, std::size_t n3)
    : dims_{n1,n2,n3,1}, total_(n1*n2*n3) {
    alloc_data_();
  }
  AlignedArray(std::size_t n1, std::size_t n2,
               std::size_t n3, std::size_t n4)
    : dims_{n1,n2,n3,n4}, total_(n1*n2*n3*n4) {
    alloc_data_();
  }

  ~AlignedArray() noexcept {
    if(data_) ::operator delete[](data_, std::align_val_t(AA_ALIGNMENT));
  }

  AlignedArray(AlignedArray const&) = delete;
  AlignedArray& operator=(AlignedArray const&) = delete;

  AlignedArray(AlignedArray&& o) noexcept
    : total_(o.total_), data_(o.data_) {
    std::copy(std::begin(o.dims_), std::end(o.dims_), std::begin(dims_));
    compute_strides_();
    adata_ = AA_ASSUME_ALIGNED_ONE_TIME(data_);
    o.data_ = nullptr;
  }
  AlignedArray& operator=(AlignedArray&& o) noexcept {
    if(this!=&o) {
      if(data_) ::operator delete[](data_, std::align_val_t(AA_ALIGNMENT));
      total_ = o.total_;
      std::copy(std::begin(o.dims_), std::end(o.dims_), std::begin(dims_));
      data_ = o.data_;
      compute_strides_();
      adata_ = AA_ASSUME_ALIGNED_ONE_TIME(data_);
      o.data_ = nullptr;
    }
    return *this;
  }

  // raw data pointer
  T*       raw()       noexcept { return data_; }
  T const* raw() const noexcept { return data_; }

  // operator() indexing (unchanged)
  inline T& operator()(std::size_t i) noexcept {
    return adata_[i];
  }
  inline T  operator()(std::size_t i) const noexcept {
    return adata_[i];
  }
  inline T& operator()(std::size_t i, std::size_t j) noexcept {
    return adata_[i*dims_[1] + j];
  }
  inline T  operator()(std::size_t i, std::size_t j) const noexcept {
    return adata_[i*dims_[1] + j];
  }
  inline T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
    return adata_[(i*dims_[1] + j)*dims_[2] + k];
  }
  inline T  operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
    return adata_[(i*dims_[1] + j)*dims_[2] + k];
  }
  inline T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) noexcept {
    return adata_[((i*dims_[1] + j)*dims_[2] + k)*dims_[3] + l];
  }
  inline T  operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const noexcept {
    return adata_[((i*dims_[1] + j)*dims_[2] + k)*dims_[3] + l];
  }

  //–––– Nested [][][] indexing
  struct Slice1 {
    T* ptr;
    inline T& operator[](std::size_t l) const noexcept { return ptr[l]; }
  };

  struct Slice2 {
    T* ptr;
    std::size_t d3;
    inline Slice1 operator[](std::size_t k) const noexcept {
      return Slice1{ ptr + k*d3 };
    }
  };

  struct Slice3 {
    T* ptr;
    std::size_t d2, d3;
    inline Slice2 operator[](std::size_t j) const noexcept {
      return Slice2{ ptr + j*d3, d3 };
    }
  };

  struct Slice4 {
    T* ptr;
    std::size_t d1, d2, d3;
    inline Slice3 operator[](std::size_t i) const noexcept {
      return Slice3{ ptr + i*d2*d3, d2*d3, d3 };
    }
  };

  inline Slice3 operator[](std::size_t i) noexcept {
    return Slice3{ adata_ + i*stride1_, stride2_, stride3_ };
  }

private:
  std::size_t dims_[4];
  std::size_t total_;
  T* __restrict__ data_ = nullptr;
  T* __restrict__ adata_ = nullptr;

  // strides: elements per block
  std::size_t stride1_, stride2_, stride3_;

  void compute_strides_() noexcept {
    // stride3 = dims_[3]
    stride3_ = dims_[3];
    // stride2 = dims_[2] * dims_[3]
    stride2_ = dims_[2] * stride3_;
    // stride1 = dims_[1] * dims_[2] * dims_[3]
    stride1_ = dims_[1] * stride2_;
  }

  void alloc_data_() {
    data_ = static_cast<T*>(
      ::operator new[](total_ * sizeof(T), std::align_val_t(AA_ALIGNMENT))
    );
    compute_strides_();
    adata_ = AA_ASSUME_ALIGNED_ONE_TIME(data_);
  }
};

#endif // ALIGNED_ARRAY_NESTED_HPP


#include <cstdlib>
#include <stdexcept>
#include <cstddef>
#include <memory>
#include <iostream>

template<typename T>
class PaddedGrid3D {
public:
    const std::size_t nx, ny, nz;
    const std::size_t padded_nx;
    const std::size_t alignment;
    const std::size_t pitch_y, pitch_z;

private:
    T* data_;

public:
    PaddedGrid3D(std::size_t nx_, std::size_t ny_, std::size_t nz_, std::size_t alignment_ = 64)
        : nx(nx_), ny(ny_), nz(nz_), alignment(alignment_),
          padded_nx(((nx + (alignment_ / sizeof(T)) - 1) / (alignment_ / sizeof(T))) * (alignment_ / sizeof(T))),
          pitch_y(padded_nx), pitch_z(pitch_y * ny), data_(nullptr)
    {
        std::size_t total_size = pitch_z * nz;
        data_ = static_cast<T*>(std::aligned_alloc(alignment, total_size * sizeof(T)));
        if (!data_) throw std::bad_alloc();
    }

    ~PaddedGrid3D() {
        std::free(data_);
    }

    inline T& operator()(std::size_t i, std::size_t j, std::size_t k) {
        return data_[(k * pitch_z) + (j * pitch_y) + i];
    }

    inline const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        return data_[(k * pitch_z) + (j * pitch_y) + i];
    }

    inline T* raw() { return data_; }
    inline const T* raw() const { return data_; }

    inline std::size_t paddedSize() const { return pitch_z * nz; }
    inline std::size_t strideX() const { return 1; }
    inline std::size_t strideY() const { return pitch_y; }
    inline std::size_t strideZ() const { return pitch_z; }
};
