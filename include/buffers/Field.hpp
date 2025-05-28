#pragma once
#include "MirrorHostDeviceBufferSH.hpp"
#include "HostPinnedBufferSH.hpp"
#include <cuda_runtime.h>
#include <type_traits>

template<typename T, uint Dim, bool hostOnly=false, bool unified=false>
class Field {
  using BufferType = std::conditional_t<hostOnly,
    HostPinnedBuffer<T,Dim,unified>,
    MirrorHostDeviceBuffer<T,Dim,unified>>;
  
public:
  Field(uint n1, uint n2=1, uint n3=1, uint n4=1)
  {
    buf_ = new BufferType(n1,n2,n3,n4);
  }
  ~Field() {
    delete buf_;
  }

  inline HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr(){
    if constexpr(hostOnly){return this->buf_; }
    else{return this->buf_->getHostBufferPtr();}
  }
  inline const HostPinnedBuffer<T,Dim,unified>* getHostBufferPtr() const noexcept {
    if constexpr(hostOnly){return this->buf_; }
    else{return this->buf_->getHostBufferPtr();}
  }
  
  inline T*       getHostDataPtr()       noexcept {
    if constexpr(hostOnly){return this->buf_->getDataPtr(); }
    else{return this->buf_->getHostDataPtr();}
  }
  inline const T* getHostDataPtr() const noexcept {
    if constexpr(hostOnly){return this->buf_->getDataPtr(); }
    else{return this->buf_->getHostDataPtr();}
  }

  
  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  inline DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() noexcept{
    return this->buf_->getDeviceBufferPtr();
  }
  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  inline const DeviceBuffer<T,Dim,unified>* getDeviceBufferPtr() const noexcept {
    return this->buf_->getDeviceBufferPtr();
  }

  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  inline T*       getDeviceDataPtr()       noexcept { return this->buf_->getDeviceDataPtr(); }
  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  inline const T* getDeviceDataPtr() const noexcept { return this->buf_->getDeviceDataPtr(); }
  
  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  void copyHostToDevice(cudaStream_t stream = 0) {
    buf_->copyHostToDevice(stream);
  }
  template<bool H = hostOnly, typename = std::enable_if_t<!H>>
  void copyDeviceToHost(cudaStream_t stream=0) {
    buf_->copyDeviceToHost(stream);
  }

  uint size() const noexcept { return buf_->size(); }
  
  template<typename... Args>
  inline T& operator()(Args... args) {
    return (*(this->getHostBufferPtr()))(args...);
  }
  template<typename... Args>
  inline const T& operator()(Args... args) const {
    return (*(this->getHostBufferPtr()))(args...);
  }

private:
  BufferType* buf_ = nullptr;
};
