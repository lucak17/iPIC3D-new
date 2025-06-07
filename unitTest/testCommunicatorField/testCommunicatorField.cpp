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
    int d1 = 10;
    int d2 = 9;
    int d3  = 8;
    int h1 = 2;
    int h2 = 2;
    int h3 = 3;
    Field<float,3,true,false> ff(d1,d2,d3,1,h1,h2,h3);
    std::cout << "In test communication rank "<< MPIManager::getInstance().getFieldRank() << std::endl;
    ff.mpiFillHaloCommunicateWaitAll<FACES>();
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
    std::cout<< "MPI topology "<< dimensions[0] << ", " << dimensions[1]  << ", " << dimensions[2] <<std::endl
;    bool periodic[Dim] = {0,0,0};
    MPIManager::getInstance().initCartesianCommunicatorField(Dim,dimensions,periodic);
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
    int b = 0;
    for(int i= 0; i< 1000; i++){
        b=b+i;
    }
    MPI_Barrier(MPIManager::getInstance().getGlobalComm());
    std::cout << "before test communication rank "<< MPIManager::getInstance().getGlobalRank() << std::endl;
    test_communication();
    std::cout << " Test communication passed" << b << std::endl;

    MPIManager::getInstance().finalize_mpi();

    return 0;
}
