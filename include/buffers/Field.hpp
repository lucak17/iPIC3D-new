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
    extentsNoHalo_ = {n1,n2,n3,n4};
    halo_ = {halo1,halo2,halo3};
    extentsWithHalo_ = {n1+2*halo1,n2+2*halo2,n3+2*halo3,n4};
    totElementsNoHalo_ = extentsNoHalo_[0] * extentsNoHalo_[1] * extentsNoHalo_[2] * extentsNoHalo_[3];
    totElementsWithHalo_ = extentsWithHalo_[0] * extentsWithHalo_[1] * extentsWithHalo_[2] * extentsWithHalo_[3]; 
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

  // copy border to halo self
  template<int Mask>
  inline int copyBorderToHaloSelf(){
    return this->mpiCommunicatorField_->template copyBorderToHaloSelf<Mask>(this->getHostDataPtr());
  }
  // copy halo to border self
  template<int Mask>
  inline int copyHaloToBorderSelf(){
    return this->mpiCommunicatorField_->template copyHaloToBorderSelf<Mask>(this->getHostDataPtr());
  }

  // fill host buffer with fixed value
  void fillHostBufferWithHalo(T value){
    std::fill_n(this->getHostDataPtr(), totElementsWithHalo_, value);
  }

  template<uint D = Dim>
  std::enable_if_t<D==3, void>
  fillIndexNoHalo(T value=1){
    for(int k = halo_[2]; k < extentsNoHalo_[2]+halo_[2]; k++){
      for(int j = halo_[1]; j < extentsNoHalo_[1]+halo_[1]; j++){
        for(int i = halo_[0]; i < extentsNoHalo_[0]+halo_[0]; i++){
          (*(this->getHostBufferPtr()))(i,j,k) = this->get1DFlatIndex(i,j,k) * value;
        }
      } 
    }
  }
  template<uint D = Dim>
  std::enable_if_t<D==3, void>
  fillIndexWithHalo(T value=1){
    for(int k = 0; k < extentsWithHalo_[2]; k++){
      for(int j = 0; j < extentsWithHalo_[1]; j++){
        for(int i = 0; i < extentsWithHalo_[0]; i++){
          (*(this->getHostBufferPtr()))(i,j,k) = this->get1DFlatIndex(i,j,k) * value;
        }
      } 
    }
  }


  template<uint D = Dim>
  std::enable_if_t<D==3, void>
  printNoHalo() const {
    for(int k = halo_[2]; k < extentsNoHalo_[2]+halo_[2]; k++){
      for(int j = halo_[1]; j < extentsNoHalo_[1]+halo_[1]; j++){
        for(int i = halo_[0]; i < extentsNoHalo_[0]+halo_[0]; i++){
          std::cout<< (*(this->getHostBufferPtr()))(i,j,k) << " ";
        }
        std::cout << "\n";
      } 
      std::cout << "\n\n";
    }
  }

  template<uint D = Dim>
  std::enable_if_t<D==3, void>
  printWithHalo() const {
    for(int k = 0; k < extentsWithHalo_[2]; k++){
      for(int j = 0; j < extentsWithHalo_[1]; j++){
        for(int i = 0; i < extentsWithHalo_[0]; i++){
          std::cout<< (*(this->getHostBufferPtr()))(i,j,k) << " ";
        }
        std::cout << "\n";
      } 
      std::cout << "\n\n";
    }
  }

  // get extents
  inline std::array<int,4> getExtentsNoHalo() const { return extentsNoHalo_;}
  inline std::array<int,4> getExtentsWithHalo() const { return extentsWithHalo_;}
  inline std::array<int,3> getHalo() const { return halo_;}

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

private:
  MPICommunicatorField<T,Dim,unified>* mpiCommunicatorField_ = nullptr;
  BufferType* buf_ = nullptr;
  std::array<int,4> extentsNoHalo_;
  std::array<int,3> halo_;
  std::array<int,4> extentsWithHalo_;
  int totElementsNoHalo_;
  int totElementsWithHalo_;
};
