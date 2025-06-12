// tests/test_Field_manual.cpp

#include <iostream>
#include <cassert>
#include "Field.hpp"
#include "MPIManager.hpp"

void test_constructor_and_size_1D() {
    std::cout << "Start test 1" << std::endl;
    Field<int,1,true,false> f(10);
    assert(f.size() == 10u);
}

void test_data_ptr_non_null_1D() {
    std::cout << "Start test 2" << std::endl;
    Field<double,1,true,false> f(5);
    assert(f.getHostDataPtr() != nullptr);
}

void test_read_write_1D() {
    Field<int,1,true,false> f(4);
    for (unsigned i = 0; i < f.size(); ++i) {
        f(i) = static_cast<int>(i * 7);
    }
    for (unsigned i = 0; i < f.size(); ++i) {
        assert(f(i) == static_cast<int>(i * 7));
    }
}

void test_const_accessors_1D() {
    Field<int,1,true,false> f(3);
    f(0) = 11; f(1) = 22; f(2) = 33;
    const auto& cf = f;
    // operator()
    assert(cf(0) == 11);
    assert(cf(2) == 33);
    // raw data pointer
    const int* raw = cf.getHostDataPtr();
    assert(raw[1] == 22);
}

void test_indexing_2D() {
    const unsigned N1 = 3, N2 = 4;
    Field<float,2,true,false> f(N1, N2);
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            float v = float(i*100 + j);
            f(i,j) = v;
        }
    }
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            float expected = float(i*100 + j);
            assert(f(i,j) == expected);
        }
    }
}

void test_communication(){
    const int myrank = MPIManager::getInstance().getFieldRank();
    int d1 = 6;
    int d2 = 6;
    int d3  = 6;
    int h1 = 2;
    int h2 = 2;
    int h3 = 2;
    Field<float,3,true,false> ff(d1,d2,d3,1,h1,h2,h3);
    auto extentsWithHalo = ff.getExtentsWithHalo();
    auto extentNoHalo = ff.getExtentsNoHalo();
    auto halo = ff.getHalo();
    std::cout << "In test communication rank "<< MPIManager::getInstance().getFieldRank() << 
    " Extents with halo " << extentsWithHalo[0] << " " << extentsWithHalo[1] << " " << extentsWithHalo[2] << std::endl;
    
    ff.fillHostBufferWithHalo(MPIManager::getInstance().getFieldRank());
    for(int k = 0; k<extentsWithHalo[2]; k++){
        for(int j = 0; j<extentsWithHalo[1]; j++){
            for(int i = 0; i<extentsWithHalo[0]; i++){
                if (ff(i,j,k) != myrank) {
                    std::cerr << "Assertion failed at ("<<i<<","<<j<<","<<k <<"): ff = "<< ff(i,j,k) <<", expected " << myrank << std::endl;
                    std::abort();
                }
            }  
        }
    }
    ff.fillIndexNoHalo();

    for(int i = 0; i < MPIManager::getInstance().getFieldNprocesses(); i++ ){
        if (myrank==i){
            std::cout<< " Field rank " << myrank<<std::endl;
            ff.printNoHalo();
            ff.printWithHalo();
        }
        MPI_Barrier(MPIManager::getInstance().getFieldComm());
    }
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    ff.mpiFillHaloCommunicateWaitAll<FACES>();
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    for(int i = 0; i < MPIManager::getInstance().getFieldNprocesses(); i++ ){
        if (myrank==i){
            std::cout<< " Field rank " << myrank<<std::endl;
            ff.printNoHalo();
            ff.printWithHalo();
        }
        MPI_Barrier(MPIManager::getInstance().getFieldComm());
    }
    int cc = ff.copyHaloToBorderSelf<FACES+EDGES+CORNERS>();
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    for(int i = 0; i < MPIManager::getInstance().getFieldNprocesses(); i++ ){
        if (myrank==i){
            std::cout<< " Field rank " << myrank<<std::endl;
            ff.printNoHalo();
            ff.printWithHalo();
        }
        MPI_Barrier(MPIManager::getInstance().getFieldComm());
    }
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    
    /*
    for(int k = 0; k<extentsWithHalo[2]; k++){
        for(int j = 0; j<extentsWithHalo[1]; j++){
            for(int i = 0; i<extentsWithHalo[0]; i++){
                std::cout<< " Field rank " << myrank << " value " << ff(i,j,k) <<std::endl;
                //if (ff(i,j,k) != myrank) {
                //    std::cerr << "Assertion failed at ("<<i<<","<<j<<","<<k <<"): ff = "<< ff(i,j,k) <<", expected " << myrank << std::endl;
                //    std::abort();
                //}
            }  
        }
    }
    */
    
    std::cout<< " count copies " << cc <<std::endl;
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    for(int k = halo[2]; k<(extentsWithHalo[2] - halo[2]); k++){
        for(int j = halo[1]; j<(extentsWithHalo[1] - halo[1]); j++){
            for(int i = halo[0]; i<(extentsWithHalo[0] - halo[0]); i++){
                std::cout<< " Field rank " << myrank << " value " << ff(i,j,k) <<std::endl;
                if (ff(i,j,k) != myrank) {
                    std::cerr << "Assertion failed at ("<<i<<","<<j<<","<<k <<"): ff = "<< ff(i,j,k) <<", expected " << myrank << std::endl;
                    std::abort();
                }
            }  
        }
    }
#if 0
    ff.mpiFillHaloCommunicateWaitAll<FACES+EDGES+CORNERS>();
    ff.copyHaloToBorderSelf<FACES+EDGES+CORNERS>();
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    for(int k = 0; k<extentsWithHalo[2]; k++){
        for(int j = 0; j<extentsWithHalo[1]; j++){
            for(int i = 0; i<extentsWithHalo[0]; i++){
                if (ff(i,j,k) != myrank) {
                    std::cerr << "Assertion failed at ("<<i<<","<<j<<","<<k <<"): ff = "<< ff(i,j,k) <<", expected " << myrank << std::endl;
                    std::abort();
                }
            }  
        }
    }
#endif

}

int main(int argc, char **argv) {

    std::cout << " Start " << std::endl;

    MPIManager::initGlobal(&argc, &argv);

    std::cout << " Start after MPI init" << std::endl;
    constexpr int Dim = 3;
    //std::array<int,Dim> dimensionsArr;
    //dimensionsArr = {std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])};
    //auto& mpiInstance = MPIManager::getInstance();
    int dimensions[Dim] = {std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])};
    std::cout<< "MPI topology "<< dimensions[0] << ", " << dimensions[1]  << ", " << dimensions[2] <<std::endl;
    bool periodic[Dim] = {1,0,0};
    MPIManager::getInstance().initCartesianCommunicatorField(Dim,dimensions,periodic);
    /*
    try {
        test_constructor_and_size_1D();
        test_data_ptr_non_null_1D();
        test_read_write_1D();
        test_const_accessors_1D();
        test_indexing_2D();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception was thrown during tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "✅ All Field manual tests passed.\n";
    */
    auto dimCommFields =  MPIManager::getInstance().getDimensionsField();
    std::cout<< "MPI field topology "<< dimCommFields[0] << ", " << dimCommFields[1]  << ", " << dimCommFields[2] <<std::endl;
    MPI_Barrier(MPIManager::getInstance().getGlobalComm());
    std::cout << "before test communication rank "<< MPIManager::getInstance().getGlobalRank() << std::endl;
    const int NtestComm = 1;
    for(int i = 0; i<NtestComm; i++){
//        test_communication();
        try {
        test_communication();
        } catch (const MPIException& e) {
        std::cerr << "Rank "
                    << MPIManager::getInstance().getFieldRank()
                    << " MPIException: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << " Test communication passed " << i+1 << "/"<<NtestComm << std::endl;
        MPI_Barrier(MPIManager::getInstance().getFieldComm());
    }
    
    int b = 0;
    for(int i= 0; i< 10000; i++){
        b=b+i;
    }
    MPI_Barrier(MPIManager::getInstance().getFieldComm());
    std::cout << " Test communication passed END ALL tests " << b << std::endl;
    MPIManager::getInstance().finalize_mpi();
    std::cout << " Test communication passed END ALL tests after finalize " << b+3 << std::endl;
    return 0;
}
