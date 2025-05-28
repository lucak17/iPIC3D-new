#include <iostream>
#include <memory>
#include <chrono>
#include <cstddef>
#include <new>       // for aligned operator new/delete


#define USE_ALIGNED_ALLOC 1

#ifndef ALIGNMENT
#define ALIGNMENT 16
#endif

#if USE_ALIGNED_ALLOC

  // – allocate N Ts on ALIGNMENT‐byte boundary
  #define ALLOC_ARRAY(T, N) \
    reinterpret_cast<T*>(::operator new[](sizeof(T)*(N), std::align_val_t(ALIGNMENT)))

  // – free what ALLOC_ARRAY gave you
  #define FREE_ARRAY(P) \
    ::operator delete[](P, std::align_val_t(ALIGNMENT))

    #define ASSUME_ALIGNED(PTR) \
    reinterpret_cast<decltype(PTR)>(__builtin_assume_aligned((PTR), ALIGNMENT))
#else
  // – plain old new[]/delete[]
  #define ALLOC_ARRAY(T, N) \
    new T[(N)]

  #define FREE_ARRAY(P) \
    delete[] (P)

    #define ASSUME_ALIGNED(PTR) \
    PTR
#endif

constexpr std::size_t N = 703;
constexpr std::size_t N_TEST = 10;      // how many times to repeat
using T_data = float;

//--------------------------------------------------------------------------------
// Flat 3D buffer: one contiguous aligned block, indexed by (i,j,k)
//--------------------------------------------------------------------------------
template<typename T>
class FlatBuffer3D {
public:
    FlatBuffer3D(std::size_t N)
      : N(N), 
        data(ALLOC_ARRAY(T, N*N*N))
    {}

    ~FlatBuffer3D() {
        FREE_ARRAY(data);
    }

    // disable copy, allow move
    FlatBuffer3D(FlatBuffer3D const&) = delete;
    FlatBuffer3D& operator=(FlatBuffer3D const&) = delete;
    FlatBuffer3D(FlatBuffer3D&& o) noexcept
      : N(o.N), data(o.data) { o.data = nullptr; }
    FlatBuffer3D& operator=(FlatBuffer3D&& o) noexcept {
        if (this != &o) {
            FREE_ARRAY(data);
            N = o.N; data = o.data;
            o.data = nullptr;
        }
        return *this;
    }

    inline T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return data[(i*N + j)*N + k];
    }
    inline T  operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data[(i*N + j)*N + k];
    }

    T* raw() noexcept { return data; }
    const T* raw() const noexcept { return data; }

private:
    std::size_t N;
    T*           data;
};

//--------------------------------------------------------------------------------
// Nested 3D buffer: array of pointers to pointers to aligned rows
//--------------------------------------------------------------------------------
template<typename T>
class NestedBuffer3D {
public:
    NestedBuffer3D(std::size_t N)
      : N(N)
    {
        // allocate top‐level pointer array
        data = new T**[N];
        // for each i, allocate array of T* pointers
        for(std::size_t i = 0; i < N; ++i) {
            data[i] = new T*[N];
            // each row (length N) is its own aligned block
            for(std::size_t j = 0; j < N; ++j) {
                data[i][j] = ALLOC_ARRAY(T, N);
            }
        }
    }

    ~NestedBuffer3D() {
        for(std::size_t i = 0; i < N; ++i) {
            for(std::size_t j = 0; j < N; ++j) {
                FREE_ARRAY(data[i][j]);
            }
            delete[] data[i];
        }
        delete[] data;
    }

    // disable copy, allow move
    NestedBuffer3D(NestedBuffer3D const&) = delete;
    NestedBuffer3D& operator=(NestedBuffer3D const&) = delete;
    NestedBuffer3D(NestedBuffer3D&& o) noexcept
      : N(o.N), data(o.data) { o.data = nullptr; }
    NestedBuffer3D& operator=(NestedBuffer3D&& o) noexcept {
        if (this != &o) {
            // free existing
            for(std::size_t i = 0; i < N; ++i) {
                for(std::size_t j = 0; j < N; ++j)
                    FREE_ARRAY(data);
                delete[] data[i];
            }
            delete[] data;
            // steal
            N    = o.N;
            data = o.data;
            o.data = nullptr;
        }
        return *this;
    }

    inline T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return data[i][j][k];
    }
    inline T  operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data[i][j][k];
    }

private:
    std::size_t N;
    T***         data;
};

//--------------------------------------------------------------------------------
// Benchmark: one 7-point Laplace sweep on each buffer, time with chrono
//--------------------------------------------------------------------------------

int main() {

    using clk = std::chrono::high_resolution_clock;

    double sum_flat = 0.0, sum_flat_assumed_aligned=0.0, sum_nest = 0.0;

    // allocate once
    FlatBuffer3D<T_data>   flat(N), flat_out(N);
    NestedBuffer3D<T_data> nest(N), nest_out(N);

#if 1
    auto *f    = ASSUME_ALIGNED(flat.raw());
    auto *fout = ASSUME_ALIGNED(flat_out.raw());
#else
    auto *f = std::assume_aligned<ALIGNMENT>(flat.raw());
    auto *fout = std::assume_aligned<ALIGNMENT>(flat_out.raw());
#endif
    // initialize once
    for(std::size_t i = 0; i < N; ++i) {
        T_data extra = T_data(i > 0 ? 10 % i : 0);
        for(std::size_t j = 0; j < N; ++j) {
            for(std::size_t k = 0; k < N; ++k) {
                T_data v = T_data(i + j + k);
                flat    (i,j,k) = v + extra;
                flat_out(i,j,k) = 0;
                nest    (i,j,k) = v + extra;
                nest_out(i,j,k)= 0;
            }
        }
    }

    // run the test N_TEST times
    for(std::size_t t = 0; t < N_TEST; ++t) {
        // --- flat buffer timing ---
        auto t0 = clk::now();
        for(std::size_t i = 1; i + 1 < N; ++i) {
          for(std::size_t j = 1; j + 1 < N; ++j) {
            for(std::size_t k = 1; k + 1 < N; ++k) {
              flat_out(i,j,k)
                = flat(i-1,j,k) + flat(i+1,j,k)
                + flat(i,j-1,k) + flat(i,j+1,k)
                + flat(i,j,k-1) + flat(i,j,k+1)
                - T_data(6) * flat(i,j,k);
            }
          }
        }
        auto t1 = clk::now();
        sum_flat += std::chrono::duration<double,std::milli>(t1 - t0).count();
        std::cout<< "Flat 1D: " << flat_out(5,5,5) <<std::endl;
    }

    for(std::size_t t = 0; t < N_TEST; ++t) {
        auto t0 = clk::now();
        for (std::size_t i = 1; i + 1 < N; ++i) {
            for (std::size_t j = 1; j + 1 < N; ++j) {
                std::size_t base = (i*N + j)*N;
                for (std::size_t k = 1; k + 1 < N; ++k) {
                    std::size_t idx = base + k;
                    fout[idx]
                        =  f[idx- N   ] +  f[idx+ N   ]   // i-1, i+1
                        +  f[idx-1    ] +  f[idx+1    ]   // j-1, j+1
                        +  f[idx-N*N  ] +  f[idx+N*N  ]   // k-1, k+1
                        - T_data(6) * f[idx];
                }
            }
        }
        auto t1 = clk::now();
        sum_flat_assumed_aligned += std::chrono::duration<double,std::milli>(t1 - t0).count();
        std::cout<< "Flat 1D assumedaligned: " << fout[(5*N + 5)*N + 5] <<std::endl;
    }

    for(std::size_t t = 0; t < N_TEST; ++t) {
        // --- nested buffer timing ---
        auto t2 = clk::now();
        for(std::size_t i = 1; i + 1 < N; ++i) {
          for(std::size_t j = 1; j + 1 < N; ++j) {
            for(std::size_t k = 1; k + 1 < N; ++k) {
              nest_out(i,j,k)
                = nest(i-1,j,k) + nest(i+1,j,k)
                + nest(i,j-1,k) + nest(i,j+1,k)
                + nest(i,j,k-1) + nest(i,j,k+1)
                - T_data(6) * nest(i,j,k);
            }
          }
        }
        auto t3 = clk::now();
        sum_nest += std::chrono::duration<double,std::milli>(t3 - t2).count();
        std::cout<< "Nested: " << nest_out(5,5,5) <<std::endl;
    }

    // compute averages
    double avg_flat = sum_flat / double(N_TEST);
    double avg_flat_assumed_aligned = sum_flat_assumed_aligned / double(N_TEST);
    double avg_nest = sum_nest / double(N_TEST);

    std::cout
      << "Ran " << N_TEST << " trials\n"
      << "Average flat buffer time:   " << avg_flat << " ms\n"
      << "Average flat assumed aligned buffer time:   " << avg_flat_assumed_aligned << " ms\n"
      << "Average nested buffer time: " << avg_nest << " ms\n";

    return 0;
}



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