#pragma once
#include "mirrorHostDeviceBufferSH.hpp"
#include <cuda_runtime.h>

template<typename T, uint Dim,bool unified=false>
class Field {
public:
  Field(uint n1, uint n2=1, uint n3=1, uint n4=1)
  {
    mirrorBuf_ = new MirrorHostDeviceBuffer<T,Dim,unified>(n1,n2,n3,n4);
  }
  ~Field() {
    delete mirrorBuf_;
  }

  inline HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr(){
    return this->mirrorBuf_->getHostBufferPtr();
  }
  inline const HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr() const noexcept {
    return this->mirrorBuf_->getHostBufferPtr();
  }

  inline DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() noexcept{
    return this->mirrorBuf_->getDeviceBufferPtr();
  }
  inline const DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() const noexcept {
    return this->mirrorBuf_->getDeviceBufferPtr();
  }

  inline T*       getHostDataPtr()       noexcept { return this->mirrorBuf_->getHostDataPtr(); }
  inline const T* getHostDataPtr() const noexcept { return this->mirrorBuf_->getHostDataPtr(); }

  inline T*       getDeviceDataPtr()       noexcept { return this->mirrorBuf_->getDeviceDataPtr(); }
  inline const T* getDeviceDataPtr() const noexcept { return this->mirrorBuf_->getDeviceDataPtr(); }

  void copyHostToDevice(cudaStream_t stream = 0) {
    mirrorBuf_->copyHostToDevice(stream);
  }
  void copyDeviceToHost(cudaStream_t stream=0) {
    mirrorBuf_->copyDeviceToHost(stream);
  }

  uint size() const noexcept { return mirrorBuf_->size(); }

  template<typename... Args>
  inline T& operator()(Args... args) {
    return (*(this->getHostBufferPtr()))(args...);
  }
  template<typename... Args>
  inline const T& operator()(Args... args) const {
    return (*(this->getHostBufferPtr()))(args...);
  }

private:
  MirrorHostDeviceBuffer<T,Dim,unified>* mirrorBuf_ = nullptr;
};
