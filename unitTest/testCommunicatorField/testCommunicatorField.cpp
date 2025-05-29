// tests/test_FieldCPU_manual.cpp

#include <iostream>
#include <cassert>
#include "FieldCPU.hpp"

void test_constructor_and_size_1D() {
    FieldCPU<int,1> f(10);
    assert(f.size() == 10u);
}

void test_data_ptr_non_null_1D() {
    FieldCPU<double,1> f(5);
    assert(f.getHostDataPtr() != nullptr);
}

void test_read_write_1D() {
    FieldCPU<int,1> f(4);
    for (unsigned i = 0; i < f.size(); ++i) {
        f(i) = static_cast<int>(i * 7);
    }
    for (unsigned i = 0; i < f.size(); ++i) {
        assert(f(i) == static_cast<int>(i * 7));
    }
}

void test_const_accessors_1D() {
    FieldCPU<int,1> f(3);
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
    FieldCPU<float,2> f(N1, N2);
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

int main() {
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

    std::cout << "✅ All FieldCPU manual tests passed.\n";
    return 0;
}
