#pragma once

#include <cuda_runtime.h>
#include <new>
#include <cstddef>
#include <stdexcept>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      throw std::runtime_error(cudaGetErrorString(err));                      \
    }                                                                         \
  } while (0)

//------------------------------------------------------------------------------
// Abstract base buffer for up to 4 dimensions
// Stores data in a flat 1D array, but provides N-dimensional access
template<typename T, std::size_t Dim>
class VirtualBuffer {
    static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4");
public:
    VirtualBuffer(std::size_t n1, std::size_t n2 = 1,
                  std::size_t n3 = 1, std::size_t n4 = 1)
      : n1_(n1), n2_(n2), n3_(n3), n4_(n4),
        total_(n1_ * n2_ * n3_ * n4_), data_(nullptr)
    {
        allocate(total_);
    }

    virtual ~VirtualBuffer() {
        deallocate();
    }

    // Raw pointer access
    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    // Total elements
    std::size_t size() const noexcept { return total_; }

     // 1D access (only enabled if Dim==1)
    template<std::size_t D = Dim>
    std::enable_if_t<D == 1, T&>
    operator()(std::size_t i) noexcept {
        return data_[i];
    }
    template<std::size_t D = Dim>
    std::enable_if_t<D == 1, const T&>
    operator()(std::size_t i) const noexcept {
        return data_[i];
    }

    // 2D access (only enabled if Dim==2)
    template<std::size_t D = Dim>
    std::enable_if_t<D == 2, T&>
    operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * n2_ + j];
    }
    template<std::size_t D = Dim>
    std::enable_if_t<D == 2, const T&>
    operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * n2_ + j];
    }

    // 3D access (only enabled if Dim==3)
    template<std::size_t D = Dim>
    std::enable_if_t<D == 3, T&>
    operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return data_[(i * n2_ + j) * n3_ + k];
    }
    template<std::size_t D = Dim>
    std::enable_if_t<D == 3, const T&>
    operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data_[(i * n2_ + j) * n3_ + k];
    }

    // 4D access (only enabled if Dim==4)
    template<std::size_t D = Dim>
    std::enable_if_t<D == 4, T&>
    operator()(std::size_t i, std::size_t j,
               std::size_t k, std::size_t l) noexcept {
        return data_[((i * n2_ + j) * n3_ + k) * n4_ + l];
    }
    template<std::size_t D = Dim>
    std::enable_if_t<D == 4, const T&>
    operator()(std::size_t i, std::size_t j,
               std::size_t k, std::size_t l) const noexcept {
        return data_[((i * n2_ + j) * n3_ + k) * n4_ + l];
    }


protected:
    virtual void allocate(std::size_t total) = 0;
    virtual void deallocate() = 0;

    std::size_t n1_, n2_, n3_, n4_, total_;
    T*           data_;
};

//------------------------------------------------------------------------------
// Host buffer: pinned or unified-aligned host memory
template<typename T, std::size_t Dim, bool Unified=false>
class HostBuffer : public VirtualBuffer<T, Dim> {
public:
    using Base = VirtualBuffer<T, Dim>;
    HostBuffer(std::size_t n1, std::size_t n2 = 1,
               std::size_t n3 = 1, std::size_t n4 = 1)
      : Base(n1, n2, n3, n4) {}

    ~HostBuffer() override = default;

protected:
    void allocate(std::size_t total) override {
        if constexpr (Unified) {
            // Aligned to 4096 bytes for unified
            this->data_ = static_cast<T*>(
                ::operator new(total * sizeof(T), std::align_val_t(4096)));
        } else {
            CUDA_CHECK(cudaHostAlloc(&this->data_,
                                     total * sizeof(T),
                                     cudaHostAllocDefault));
        }
    }
    void deallocate() override {
        if constexpr (Unified) {
            ::operator delete(this->data_, std::align_val_t(4096));
        } else {
            if (this->data_) cudaFreeHost(this->data_);
        }
    }
};

//------------------------------------------------------------------------------
// Device buffer: GPU global or unified-aligned memory
template<typename T, std::size_t Dim, bool Unified=false>
class DeviceBuffer : public VirtualBuffer<T, Dim> {
public:
    using Base = VirtualBuffer<T, Dim>;
    DeviceBuffer(std::size_t n1, std::size_t n2 = 1,
                 std::size_t n3 = 1, std::size_t n4 = 1)
      : Base(n1, n2, n3, n4) {}

    ~DeviceBuffer() override = default;

protected:
    void allocate(std::size_t total) override {
        if constexpr (Unified) {
            // Use host-aligned allocation for device too
            this->data_ = static_cast<T*>(
                ::operator new(total * sizeof(T), std::align_val_t(4096)));
        } else {
            CUDA_CHECK(cudaMalloc(&this->data_, total * sizeof(T)));
        }
    }
    void deallocate() override {
        if constexpr (Unified) {
            ::operator delete(this->data_, std::align_val_t(4096));
        } else {
            if (this->data_) cudaFree(this->data_);
        }
    }
};

//------------------------------------------------------------------------------
// Field abstraction: choose unified or separate at compile time
template<typename T, std::size_t Dim, bool Unified = false>
class Field {
public:
    Field(std::size_t n1, std::size_t n2 = 1,
          std::size_t n3 = 1, std::size_t n4 = 1) {
        if constexpr (Unified) {
            hostBuf_ = new HostBuffer<T, Dim, true>(n1, n2, n3, n4);
            deviceBuf_ = hostBuf_;
        } else {
            hostBuf_ = new HostBuffer<T, Dim, false>(n1, n2, n3, n4);
            deviceBuf_ = new DeviceBuffer<T, Dim, false>(n1, n2, n3, n4);
        }
    }

    ~Field() {
        if constexpr (Unified) {
            delete hostBuf_;
        } else {
            delete hostBuf_;
            delete deviceBuf_;
        }
    }

    // Host pointer
    T*       hostData()       noexcept { return hostBuf_->data(); }
    const T* hostData() const noexcept { return hostBuf_->data(); }

    // Device pointer
    T*       deviceData()       noexcept { return deviceBuf_->data(); }
    const T* deviceData() const noexcept { return deviceBuf_->data(); }

    // Copy H->D
    void copyToDevice(cudaStream_t s = 0) {
        if constexpr (!Unified) {
            CUDA_CHECK(cudaMemcpyAsync(deviceBuf_->data(),
                                       hostBuf_->data(),
                                       hostBuf_->size() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       s));
        }
    }
    // Copy D->H
    void copyToHost(cudaStream_t s = 0) {
        if constexpr (!Unified) {
            CUDA_CHECK(cudaMemcpyAsync(hostBuf_->data(),
                                       deviceBuf_->data(),
                                       hostBuf_->size() * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       s));
        }
    }

    // N-dimensional access forwarded to host buffer
    template<typename... Args>
    T& operator()(Args... args) {
        return (*hostBuf_)(args...);
    }
    template<typename... Args>
    const T& operator()(Args... args) const {
        return (*hostBuf_)(args...);
    }

private:
    VirtualBuffer<T, Dim>* hostBuf_;
    VirtualBuffer<T, Dim>* deviceBuf_;
};
