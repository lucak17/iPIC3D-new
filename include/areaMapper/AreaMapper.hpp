#pragma once
#include <array>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <iostream>

#include <VirtualBuffer.hpp>


using uint = std::uint32_t;


// bitmask for halo‐exchange stages
enum HaloExchange : int {
    FACES   = 1<<0,
    EDGES   = 1<<1,
    CORNERS = 1<<2,
};

enum AreaMapping : int {
    CORE = 1<<0,
    BORDER = 1<<1,
    HALO = 1<<2,
};

class AreaMapper {
    
public:
    AreaMapper(){

    }
    ~AreaMapper(){

    }


private:

};