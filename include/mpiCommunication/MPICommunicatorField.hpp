#pragma once
#include <mpi.h>
#include <array>
#include <vector>
#include <tuple>
#include <stdexcept>


using uint = std::uint32_t;

// bitmask for halo‐exchange stages
enum HaloExchange : int {
    FACES   = 1<<0,
    EDGES   = 1<<1,
    CORNERS = 1<<2,
};


class CommunicatorField
{

public:
    CommunicatorField(uint haloX, uint haloY, uint haloZ)
    {
        halo[0] = haloX;
        halo[1] = haloY;
        halo[2] = haloZ;
    
    }

    ~CommunicatorField(){}


private:
    uint halo[3];
};


template<int Mask>
void communicate(double* data,
                 int nx, int ny, int nz,
                 int H,
                 MPI_Comm cart)
{
    // full sizes including halos
    int size[3] = { nz + 2*H,
                    ny + 2*H,
                    nx + 2*H };
    int coords[3];
    int me;
    MPI_Comm_rank(cart, &me);
    MPI_Cart_coords(cart, me, 3, coords);

    // helper to exchange in a single direction (dx,dy,dz ∈ { -1,0,1 })
    auto exch = [&](int dx,int dy,int dz){
        // find neighbour
        int nbr_coords[3] = { coords[0]+dz,
                              coords[1]+dy,
                              coords[2]+dx };
        int nbr;
        if (MPI_Cart_rank(cart, nbr_coords, &nbr) != MPI_SUCCESS
         || nbr == MPI_PROC_NULL)
            return;  // no neighbor in that direction

        // build subarray type for send
        int subsize[3] = {
            dz==0 ? nz      : H,  // full interior in z if dz==0, else halo thickness
            dy==0 ? ny      : H,
            dx==0 ? nx      : H
        };
        int start_send[3] = {
            dz==+1 ? nz      : (dz==0 ? H : H),   // +1: send the "top" interior face
            dy==+1 ? ny      : (dy==0 ? H : H),
            dx==+1 ? nx      : (dx==0 ? H : H)
        };
        MPI_Datatype sendtype;
        MPI_Type_create_subarray(3, size, subsize, start_send,
                                 MPI_ORDER_C, MPI_DOUBLE, &sendtype);
        MPI_Type_commit(&sendtype);

        // build subarray type for recv
        int start_recv[3] = {
            dz==+1 ? nz+H    : (dz==0 ? H : 0),   // +1: recv into top halo
            dy==+1 ? ny+H    : (dy==0 ? H : 0),
            dx==+1 ? nx+H    : (dx==0 ? H : 0)
        };
        MPI_Datatype recvtype;
        MPI_Type_create_subarray(3, size, subsize, start_recv,
                                 MPI_ORDER_C, MPI_DOUBLE, &recvtype);
        MPI_Type_commit(&recvtype);

        // post nonblocking
        MPI_Request rs, ss;
        MPI_Irecv(data, 1, recvtype, nbr, 0, cart, &rs);
        MPI_Isend(data, 1, sendtype, nbr, 0, cart, &ss);

        MPI_Wait(&rs, MPI_STATUS_IGNORE);
        MPI_Wait(&ss, MPI_STATUS_IGNORE);

        MPI_Type_free(&sendtype);
        MPI_Type_free(&recvtype);
    };

    // 1) Faces: all (dx,dy,dz) with |dx|+|dy|+|dz| == 1
    if constexpr (Mask & FACES) {
        constexpr std::array<std::tuple<int,int,int>,6> faces = {{
            {  1,  0,  0 }, { -1,  0,  0 },
            {  0,  1,  0 }, {  0, -1,  0 },
            {  0,  0,  1 }, {  0,  0, -1 }
        }};
        for (auto [dx,dy,dz] : faces) exch(dx,dy,dz);
    }

    // 2) Edges:   |dx|+|dy|+|dz| == 2
    if constexpr (Mask & EDGES) {
        constexpr std::array<std::tuple<int,int,int>,12> edges = {{
            { ±1, ±1,  0 }, { ±1,  0, ±1 }, {  0, ±1, ±1 }
        }};
        for (auto [dx,dy,dz] : edges) exch(dx,dy,dz);
    }

    // 3) Corners: |dx|+|dy|+|dz| == 3
    if constexpr (Mask & CORNERS) {
        constexpr std::array<std::tuple<int,int,int>,8> corners = {{
            {  1,  1,  1 }, {  1,  1, -1 },
            {  1, -1,  1 }, {  1, -1, -1 },
            { -1,  1,  1 }, { -1,  1, -1 },
            { -1, -1,  1 }, { -1, -1, -1 }
        }};
        for (auto [dx,dy,dz] : corners) exch(dx,dy,dz);
    }
}
