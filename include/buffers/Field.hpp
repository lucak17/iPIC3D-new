#pragma once
#include "MirrorHostDeviceBufferSH.hpp"
#include "HostPinnedBufferSH.hpp"
#include "MPICommunicatorField.hpp"
#include <cuda_runtime.h>
#include <type_traits>

using uint = std::uint32_t;

template<typename T, uint Dim, bool hostOnly=false, bool unified=false>
class Field {
  using BufferType = std::conditional_t<hostOnly,
    HostPinnedBuffer<T,Dim,unified>,
    MirrorHostDeviceBuffer<T,Dim,unified>>;
  
public:
  Field(uint n1, uint n2=1, uint n3=1, uint n4=1, uint halo1=0, uint halo2=0, uint halo3=0)
  {
    buf_ = new BufferType(n1+2*halo1,n2+2*halo2,n3+2*halo3,n4);
    extentsNoHalo = {n1,n2,n3};
    halo = {halo1,halo2,halo3};
    extents = {n1+2*halo1,n2+2*halo2,n3+2*halo3};
    mpiCommunicatorField_ = new MPICommunicatorField<T,Dim,unified>(n1,n2,n3,halo1,halo2,halo3,*this->getHostBufferPtr());
  }
  ~Field() {
    delete buf_;
    delete mpiCommunicatorField_;
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
  
  // access data
  template<typename... Args>
  inline T& operator()(Args... args) {
    return (*(this->getHostBufferPtr()))(args...);
  }
  template<typename... Args>
  inline const T& operator()(Args... args) const {
    return (*(this->getHostBufferPtr()))(args...);
  }

  template<typename... Args>
  inline uint get1DFlatIndex(Args... args) const {
    return this->getHostBufferPtr()->get1DFlatIndex(args...);
  }

  // communicate boundary -> fill halo
  template<int Mask>
  inline void mpiFillHaloCommunicateWaitAll(){
    this->mpiCommunicatorField_->template communicateFillHaloStartAndWaitAll<Mask>(this->getHostDataPtr());
  }
  template<int Mask>
  inline void mpiFillHaloCommunicateQuickReturn(){
    this->mpiCommunicatorField_->template communicateFillHaloStart<Mask>(this->getHostDataPtr());
  }
  inline void mpifillHaloWaitAll() const {
    this->mpiCommunicatorField_->communicateWaitAllSendAndCheck();
    this->mpiCommunicatorField_->communicateWaitAllRcvAndCheck();
  }


private:
  MPICommunicatorField<T,Dim,unified>* mpiCommunicatorField_ = nullptr;
  BufferType* buf_ = nullptr;
  std::array<int,3> extentsNoHalo;
  std::array<int,3> halo;
  std::array<int,3> extents;
};
