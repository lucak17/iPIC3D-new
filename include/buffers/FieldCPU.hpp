#pragma once
#include "HostBufferCPUSH.hpp"
#include <type_traits>

template<typename T, uint Dim, bool unified=false>
class FieldCPU {
public:
  FieldCPU(uint n1, uint n2=1, uint n3=1, uint n4=1)
  {
    buf_ = new HostBufferCPU<T,Dim,unified>(n1,n2,n3,n4);
  }
  ~FieldCPU() {
    delete buf_;
  }

  inline HostBufferCPU<T,Dim,unified>* getHostBufferPtr(){return this->buf_; }
  inline const HostBufferCPU<T,Dim,unified>* getHostBufferPtr() const noexcept{return this->buf_; }
  
  inline T*       getHostDataPtr()       noexcept {return this->buf_->getDataPtr(); }
  inline const T* getHostDataPtr() const noexcept {return this->buf_->getDataPtr(); }
  

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
  HostBufferCPU<T,Dim,unified>* buf_ = nullptr;
};
