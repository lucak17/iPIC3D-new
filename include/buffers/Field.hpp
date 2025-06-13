#pragma once
#include "MirrorHostDeviceBufferSH.hpp"
#include "HostPinnedBufferSH.hpp"
#include "MPICommunicatorField.hpp"
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>
#include <array>
#include <algorithm>
#include <cstddef>

using uint = std::uint32_t;


// for a given D (1…Dim) generates exactly D nested loops
template <uint D, uint Dim>
struct ForEachField {
  template <typename Func>
  static void apply(std::array<int,Dim> const& start, std::array<int,Dim> const& end, Func&& f, std::array<int,Dim>& idx){
    // Loop dimension D-1, then recurse for the next inner loop
    for(idx[D-1] = start[D-1]; idx[D-1] < end[D-1]; ++idx[D-1]){
      ForEachField<D-1,Dim>::apply(start, end, std::forward<Func>(f), idx);
    }
  }
};
// case D==0 calls the function
template <uint Dim>
struct ForEachField<0,Dim>{
  template <typename Func>
  static void apply(std::array<int,Dim> const& start, std::array<int,Dim> const& end, Func&& f, std::array<int,Dim>& idx){
    // Unpack idx[0]…idx[Dim-1] into f
    std::apply(f, idx);
  }
};


template<typename T, std::size_t Dimin, std::size_t Dimout>
constexpr std::array<T,Dimout> shrinkArray(const std::array<T,Dimin>& arrayIn) {
    static_assert(Dimout <= Dimin, "Dimout must be <= Dimin");
    std::array<T,Dimout> tmp{};
    // Copy the first Dimout elements
    std::copy_n(arrayIn.begin(), Dimout, tmp.begin());
    return tmp;
}


template<typename T, uint Dim, bool hostOnly=false, bool unified=false>
class Field {
  using BufferType = std::conditional_t<hostOnly,
    HostPinnedBuffer<T,Dim,unified>,
    MirrorHostDeviceBuffer<T,Dim,unified>>;
  
public:
  Field(uint n0, uint n1=1, uint n2=1, uint n3=1, uint halo0=0, uint halo1=0, uint halo2=0,uint halo3=0)
  {
    extentsNoHalo_ = {n0, n1, n2, n3};
    halo_ = {halo0, halo1, halo2, halo3};
    extentsWithHalo_ = {n0+2*halo0, n1+2*halo1, n2+2*halo2, n3+2*halo3};
    totElementsNoHalo_ = extentsNoHalo_[0] * extentsNoHalo_[1] * extentsNoHalo_[2] * extentsNoHalo_[3];
    totElementsWithHalo_ = extentsWithHalo_[0] * extentsWithHalo_[1] * extentsWithHalo_[2] * extentsWithHalo_[3]; 
    startNoHalo_ = shrinkArray<int,4,Dim>(std::array<int,4>{halo0, halo1, halo2, halo3});
    endNoHalo_ = shrinkArray<int,4,Dim>(std::array<int,4>{extentsNoHalo_[0]+halo_[0], extentsNoHalo_[1]+halo_[1], extentsNoHalo_[2]+halo_[2], extentsNoHalo_[3]+halo_[3]});
    startWithHalo_ = shrinkArray<int,4,Dim>(std::array<int,4>{0, 0, 0, 0});
    endWithHalo_ = shrinkArray<int,4,Dim>(std::array<int,4>{extentsWithHalo_[0], extentsWithHalo_[1], extentsWithHalo_[2], extentsWithHalo_[3]});
    buf_ = new BufferType(n0+2*halo0, n1+2*halo1, n2+2*halo2, n3+2*halo3);
    mpiCommunicatorField_ = new MPICommunicatorField<T,Dim,unified>(n0, n1, n2, n3, halo0, halo1, halo2, halo3, *this->getHostBufferPtr());
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
  template<int Mask, bool onlyCommunicationSides = true>
  inline int copyBorderToHaloSelf(){
    return this->mpiCommunicatorField_->template copyBorderToHaloSelf<Mask, onlyCommunicationSides>(this->getHostDataPtr());
  }
  // copy halo to border self
  template<int Mask, bool onlyCommunicationSides = true>
  inline int copyHaloToBorderSelf(){
    return this->mpiCommunicatorField_->template copyHaloToBorderSelf<Mask, onlyCommunicationSides>(this->getHostDataPtr());
  }

  // fill host buffer with fixed value
  void fillHostBufferWithHalo(T value){
    std::fill_n(this->getHostDataPtr(), totElementsWithHalo_, value);
  }
  
  void fillHostBufferNoHalo(T value = static_cast<T>(1)) {
    std::array<int,Dim> idx{};
    ForEachField<Dim,Dim>::apply(
      startNoHalo_, endNoHalo_, [&](auto... I){ (*this->getHostBufferPtr())(I...) = value; }, idx );
  }

  void fillIndexNoHalo(T value = static_cast<T>(1)) {
    std::array<int,Dim> idx{};
    ForEachField<Dim,Dim>::apply(
      startNoHalo_, endNoHalo_, [&](auto... I){ (*this->getHostBufferPtr())(I...) = this->get1DFlatIndex(I...) * value; }, idx );
  }
  
  void fillIndexWithHalo(T value = static_cast<T>(1)) {
    std::array<int,Dim> idx{};
    ForEachField<Dim,Dim>::apply(
      startWithHalo_, endWithHalo_, [&](auto... I){ (*this->getHostBufferPtr())(I...) = this->get1DFlatIndex(I...) * value; }, idx );
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
  inline std::array<int,4> getHalo() const { return halo_;}

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
  std::array<int,4> halo_;
  std::array<int,4> extentsWithHalo_;
  std::array<int,Dim> startNoHalo_;
  std::array<int,Dim> endNoHalo_;
  std::array<int,Dim> startWithHalo_;
  std::array<int,Dim> endWithHalo_;
  int totElementsNoHalo_;
  int totElementsWithHalo_;
};
