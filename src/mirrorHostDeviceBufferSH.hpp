#pragma once
#include "virtualBuffer.hpp"
#include "hostPinnedBufferSH.hpp"
#include "deviceBufferSH.hpp"
#include <cuda_runtime.h>
#include <new>
#include <cstring>
#include <stdexcept>



template<typename T, uint Dim, bool unified = false>
class MirrorHostDeviceBuffer : public VirtualBuffer<T,Dim,unified> {
public:
  using Base = VirtualBuffer<T,Dim,unified>;

  // ctor/dtor
    MirrorHostDeviceBuffer(uint n1, uint n2 = 1, uint n3 = 1, uint n4 = 1) : Base(n1,n2,n3,n4) 
    {
        this->allocate(n1,n2,n3,n4);
    }

    ~MirrorHostDeviceBuffer() override{
        this->deallocate();
    }

    void copyHostToDevice(cudaStream_t stream = 0){
        if constexpr (unified){}
        else{
            this->hostBuf_->copyTo(*(this->deviceBuf_), stream);
        }
    }
    
    void copyDeviceToHost(cudaStream_t stream = 0){
        if constexpr (unified){}
        else{
            this->hostBuf_->copyFrom(*(this->deviceBuf_), stream);
        }
    }

    void selfCopy(T* newPtr, T* oldPtr, uint oldSize) override {}

    void expandBuffer(const uint n1, const uint n2=1, const uint n3=1, const uint n4=1, cudaStream_t stream = 0 ) override {
        this->setExtents(n1,n2,n3,n4);
        hostBuf_->expandBuffer(n1,n2,n3,n4,stream);
        deviceBuf_->expandBuffer(n1,n2,n3,n4,stream);
  }

    inline HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr(){
        return this->hostBuf_;
    }
    inline const HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr() const noexcept {
        return this->hostBuf_;
    }

    inline DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() noexcept{
        return this->deviceBuf_;
    }
    inline const DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() const noexcept {
        return this->deviceBuf_;
    }

    __host__ __device__ __forceinline__ T*       getHostDataPtr()       noexcept { return this->hostBuf_->getDataPtr(); }
    __host__ __device__ __forceinline__ const T* getHostDataPtr() const noexcept { return this->hostBuf_->getDataPtr(); }

    __host__ __device__ __forceinline__ T* getDeviceDataPtr() noexcept { 
        if constexpr (unified){return this->hostBuf_->getDataPtr();} 
        else{return this->deviceBuf_->getDataPtr();}
    }
    __host__ __device__ __forceinline__ const T* getDeviceDataPtr() const noexcept { 
        if constexpr (unified){return this->hostBuf_->getDataPtr();} 
        else{return this->deviceBuf_->getDataPtr();}
    }

protected:
    void allocate(const uint n1, const uint n2=1, const uint n3=1, const uint n4=1) override {
        if(this->total_ > 0){
            if constexpr (unified){
                hostBuf_ = new HostPinnedBuffer<T, Dim, unified>(n1,n2,n3,n4);
                deviceBuf_ = new DeviceBuffer<T, Dim, unified>(n1,n2,n3,n4);
                deviceBuf_->allocateForUnifiedMemory(this->getHostDataPtr());
            }
            else{
                hostBuf_ = new HostPinnedBuffer<T, Dim, unified>(n1,n2,n3,n4);
                deviceBuf_ = new DeviceBuffer<T, Dim, unified>(n1,n2,n3,n4);
            }
        }
    }
    void deallocate() override {
        if (this->hostBuf_ != nullptr){ delete hostBuf_; }
        if (this->deviceBuf_ != nullptr){ delete deviceBuf_; }
    }
    void deallocatePtr(T* ptr) override {}
private:

  HostPinnedBuffer<T,Dim,unified>* hostBuf_ = nullptr;
  DeviceBuffer<T,Dim,unified>* deviceBuf_ = nullptr;
};
