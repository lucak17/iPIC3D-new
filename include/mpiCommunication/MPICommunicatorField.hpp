#pragma once
#include <mpi.h>
#include <array>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <iostream>

#include <VirtualBuffer.hpp>
#include "MPIManager.hpp"


#define NUM_COMM 26
using uint = std::uint32_t;

// bitmask for halo‐exchange stages
enum HaloExchange : int {
    FACES   = 1<<0,
    EDGES   = 1<<1,
    CORNERS = 1<<2,
};

// function to set MPI_Datatype according to T in MPI calls
template <typename T>
inline MPI_Datatype getMPIType();
template <>
inline MPI_Datatype getMPIType<float>() {
    return MPI_FLOAT;
}
template <>
inline MPI_Datatype getMPIType<double>() {
    return MPI_DOUBLE;
}
template <>
inline MPI_Datatype getMPIType<int>() {
    return MPI_INT;
}

template<typename T, uint Dim, bool unified=false>
class MPICommunicatorField
{

public:
    MPICommunicatorField(uint nx, uint ny, uint nz, uint haloX, uint haloY, uint haloZ, const VirtualBuffer<T,Dim,unified>& buffer)
    {
        halo[0] = haloX;
        halo[1] = haloY;
        halo[2] = haloZ;
        for (auto &dt : border_type) dt = MPI_DATATYPE_NULL;
        for (auto &dt : halo_type) dt = MPI_DATATYPE_NULL;
        // set all ranks to -1 --> null value
        neighbour_.fill(-1);
        requestsSend_ = new MPI_Request[NUM_COMM];
        requestsRcv_ = new MPI_Request[NUM_COMM];
        statusesSend_ = new MPI_Status[NUM_COMM];
        statusesRcv_ = new MPI_Status[NUM_COMM];
        createMPIDataType(nx,ny,nz,buffer);
        computeNeighbourRank();
    }

    ~MPICommunicatorField()
    {
        for(int i = 0; i<static_cast<int>(border_type.size()); i++){
            if (border_type[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&border_type[i]);
                border_type[i] = MPI_DATATYPE_NULL;
            }
            if (halo_type[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&halo_type[i]);
                halo_type[i] = MPI_DATATYPE_NULL;
            }
        }
        delete[] requestsSend_;
        delete[] requestsRcv_;
        delete[] statusesSend_;
        delete[] statusesRcv_;
    }

    void createMPIDataType(uint nx, uint ny, uint nz, const VirtualBuffer<T,Dim,unified>& buffer){
        constexpr uint communicationDim = 3;
        if constexpr (Dim == 3)
        {
            std::array<uint,Dim> idxStartSend;
            std::array<uint,Dim> idxEndSend;
            std::array<uint,Dim> extentNoHalo;
            extentNoHalo[0] = nx;
            extentNoHalo[1] = ny;
            extentNoHalo[2] = nz;
            // loop over the communication directions 
            // direction indexes 0, 1, 2  map to -1, 0, 1 
            for(uint dirz = 0; dirz<communicationDim; dirz++ ){
                for(uint diry = 0; diry<communicationDim; diry++ ){
                    for(uint dirx = 0; dirx<communicationDim; dirx++ ){
                        if (dirx == 1 && diry == 1 && dirz == 1) continue;
                        uint idxDirectionComm = dirx + diry * communicationDim + dirz * communicationDim * communicationDim;
                        // indexes without halo points --> 0,0,0 is first element of inner+boundary
                        std::array<int,communicationDim> dir;
                        // back to indexes -1 0 1
                        dir[0] = dirx - 1;
                        dir[1] = diry - 1;
                        dir[2] = dirz - 1;
                        uint numElementsComm = 1;
                        for(uint d = 0; d<communicationDim; d++){
                            idxStartSend[d] = dir[d] > 0 ? extentNoHalo[d] - halo[d] : 0;
                            // last data to send has idx = idxEndSend - 1
                            idxEndSend[d] = dir[d] == 0 ? extentNoHalo[d] : idxStartSend[d]+halo[d];
                            // idxStartRcv[d] = dir[d] < 0 ? idxStartSend[d] - halo[d] : ( dir[d] == 0 ? 0 : idxEndSend[d]);
                            numElementsComm *= (idxEndSend[d] - idxStartSend[d]);
                        }
                        int* block_length = new int[numElementsComm];
                        int* border_indx = new int[numElementsComm];
                        int* halo_indx = new int[numElementsComm];
                        std::fill(block_length, block_length + numElementsComm, 1);
                        uint idxLoop = 0;
                        for(uint k = idxStartSend[2]; k<idxEndSend[2]; k++){
                            for(uint j = idxStartSend[1]; j<idxEndSend[1]; j++){
                                for(uint i = idxStartSend[0]; i<idxEndSend[0]; i++){
                                    uint iShift = i + halo[0];
                                    uint jShift = j + halo[1];
                                    uint kShift = k + halo[2];
                                    border_indx[idxLoop] = buffer.get1DFlatIndex(iShift, jShift, kShift);
                                    halo_indx[idxLoop] = buffer.get1DFlatIndex(iShift + dir[0]*halo[0], jShift + dir[1]*halo[1], kShift + dir[2]*halo[2]);
                                    idxLoop++;
                                }
                            }
                        }

                        MPI_Type_indexed(numElementsComm, block_length, border_indx, getMPIType<T>(), &border_type[idxDirectionComm]);
                        MPI_Type_commit(&border_type[idxDirectionComm]);
                        MPI_Type_indexed(numElementsComm, block_length, halo_indx, getMPIType<T>(), &halo_type[idxDirectionComm]);
                        MPI_Type_commit(&halo_type[idxDirectionComm]);

                        delete[] border_indx;
                        delete[] halo_indx;
                        delete[] block_length;
                    }
                }
            }
        }
        else if (Dim == 4)
        {
            std::cerr<< " MPI field communicator for Dim=4 is not implemented yet " <<std::endl;
        }
    }

    void computeNeighbourRank(){
        
        constexpr int communicationDim = 3;
        MPI_Comm  fieldComm  = MPIManager::getInstance().getFieldComm();
        const auto myCoords = MPIManager::getInstance().getCoordsField();
        auto computeNghb = [&](int dirx,int diry,int dirz) -> void{
            const int idxDirectionComm = (dirx + 1) + (diry + 1) * communicationDim + (dirz + 1) * communicationDim * communicationDim;
            assert(idxDirectionComm != 13);
            // find neighbour
            const int neighbour_coords[3] = {myCoords[0] + dirx, myCoords[1] + diry, myCoords[2] + dirz};
            try {
                mpi_check(MPI_Cart_rank(fieldComm, neighbour_coords, &neighbour_[idxDirectionComm]));
            }
            catch (const MPIException& ex) {
                std::cerr << "Warning: MPI_Cart_rank failed for mycoords (" << myCoords[0] << "," << myCoords[1] << "," << myCoords[2] << ") " << " - neighbour_coords (" << neighbour_coords[0] << "," << neighbour_coords[1] << "," << neighbour_coords[2] << "): " << ex.what() << "\n";
                neighbour_[idxDirectionComm] = -1;
                return;  // skip this direction
            }
        };
        
        constexpr std::array<std::array<int,3>,26> directions = {{ 
            // faces
            std::array<int,3>{-1,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{0,-1,0}, 
            std::array<int,3>{0,1,0}, std::array<int,3>{0,0,-1}, std::array<int,3>{0,0,1},
            // edges 
            std::array<int,3>{-1,-1,0}, std::array<int,3>{1,-1,0}, std::array<int,3>{-1,1,0}, std::array<int,3>{1,1,0},
            std::array<int,3>{-1,0,-1}, std::array<int,3>{1,0,-1}, std::array<int,3>{-1,0,1}, std::array<int,3>{1,0,1},
            std::array<int,3>{0,-1,-1}, std::array<int,3>{0,1,-1}, std::array<int,3>{0,-1,1}, std::array<int,3>{0,1,1},
            // corners
            std::array<int,3>{-1,-1,-1}, std::array<int,3>{1,-1,-1}, std::array<int,3>{1,1,-1}, std::array<int,3>{1,1,1},
            std::array<int,3>{-1,1,-1}, std::array<int,3>{-1,-1,1}, std::array<int,3>{-1,1,1}, std::array<int,3>{1,-1,1} }};
        for (auto const& [dirx,diry,dirz] : directions){
            computeNghb(dirx,diry,dirz);
        }        
    }

    void communicateWaitAllSendAndCheck(int countMsg) const {
        mpi_check( MPI_Waitall(countMsg, requestsSend_, statusesSend_) );
        for (int i = 0; i < countMsg; i++){
            if (statusesSend_[i].MPI_ERROR != MPI_SUCCESS) {
                std::cerr << "Error in MPI send " << i << ": " << statusesSend_[i].MPI_ERROR << std::endl;
            }
        }
    }

    void communicateWaitAllRcvAndCheck(int countMsg) const {
        mpi_check( MPI_Waitall(countMsg, requestsRcv_, statusesRcv_) );
        for (int i = 0; i < countMsg; i++) {
            if (statusesRcv_[i].MPI_ERROR != MPI_SUCCESS) {
                std::cerr << "Error in MPI recv " << i << ": " << statusesRcv_[i].MPI_ERROR << std::endl;
            }
        }
    }

    template<int Mask>
    int communicateFillHaloStart(T* data){

        constexpr int communicationDim = 3;
        MPI_Comm  fieldComm  = MPIManager::getInstance().getFieldComm();
        countComm_ = 0;
        /*
        int maxNumComm = 0;
        if constexpr (Mask & FACES) {maxNumComm += 6;}
        if constexpr (Mask & EDGES) {maxNumComm += 12;}
        if constexpr (Mask & CORNERS) {maxNumComm += 8;}
        */
        int myrank = MPIManager::getInstance().getFieldRank();
        const auto myCoords = MPIManager::getInstance().getCoordsField();
        auto startSendRcv = [&](int dirx,int diry,int dirz) -> void {

            const int idxDirectionComm = (dirx + 1) + (diry + 1) * communicationDim + (dirz + 1) * communicationDim * communicationDim;
            assert(idxDirectionComm != 13);
            // idxDirectionCommRcv = idxDirectionComm of the other rank that is sending data
            const int idxDirectionCommRcv = (-1 * dirx + 1) + (-1 * diry + 1) * communicationDim + (-1 * dirz + 1) * communicationDim * communicationDim;

            // find neighbour
            const int neighbour_coords[3] = {myCoords[0] + dirx, myCoords[1] + diry, myCoords[2] + dirz};
            const int neighbour = neighbour_[idxDirectionComm];
            if(neighbour < 0)return;

            //std::cout<< "My coords " << myCoords[0] << " " << myCoords[1] << " "<< myCoords[2]
            //<< " - Ngbh coords " << neighbour_coords[0] << " " << neighbour_coords[1] << " "<< neighbour_coords[2] << " - Ngbh rank "<< neighbour <<std::endl;

            std::cout<< "In startSendRcv mycoords (" << myCoords[0] << "," << myCoords[1] << "," << myCoords[2] << ") -> rank " << myrank << " - neighbour_coords (" << neighbour_coords[0] << "," << neighbour_coords[1] << "," << neighbour_coords[2] << ") -> neighbour " << neighbour
             << " - sendDir " << "(" << dirx << "," << diry << "," << dirz << ") idxDirectionComm " 
             << idxDirectionComm << " idxDirectionCommRcv " << idxDirectionCommRcv <<std::endl;
            // nonblocking communication
#if 1
            mpi_check( MPI_Irecv(data, 1, halo_type[idxDirectionComm], neighbour, idxDirectionCommRcv, fieldComm, &requestsRcv_[countComm_]) );
            mpi_check( MPI_Isend(data, 1, border_type[idxDirectionComm], neighbour, idxDirectionComm, fieldComm, &requestsSend_[countComm_]) );
#else
            mpi_check( MPI_Irecv(data, 1, halo_type[idxDirectionComm], neighbour, 0, fieldComm, &requestsRcv_[countComm_]) );
            mpi_check( MPI_Isend(data, 1, border_type[idxDirectionComm], neighbour, 0, fieldComm, &requestsSend_[countComm_]) );
#endif
            countComm_++;
        };

        // communicate faces
        if constexpr (Mask & FACES){
            constexpr std::array<std::array<int,3>,6> faces = {{ 
                std::array<int,3>{-1,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{0,-1,0}, 
                std::array<int,3>{0,1,0}, std::array<int,3>{0,0,-1}, std::array<int,3>{0,0,1} }};
            for (auto const& [dirx,diry,dirz] : faces) startSendRcv(dirx,diry,dirz);
        }

        // communicate edges
        if constexpr (Mask & EDGES){
            constexpr std::array<std::array<int,3>,12> edges = {{ 
                std::array<int,3>{-1,-1,0}, std::array<int,3>{1,-1,0}, std::array<int,3>{-1,1,0}, std::array<int,3>{1,1,0},
                std::array<int,3>{-1,0,-1}, std::array<int,3>{1,0,-1}, std::array<int,3>{-1,0,1}, std::array<int,3>{1,0,1},
                std::array<int,3>{0,-1,-1}, std::array<int,3>{0,1,-1}, std::array<int,3>{0,-1,1}, std::array<int,3>{0,1,1} }};
            for (auto const& [dirx,diry,dirz] : edges) startSendRcv(dirx,diry,dirz);
        }
        
        // communicate corners
        if constexpr (Mask & CORNERS){
            constexpr std::array<std::array<int,3>,8> corners = {{ 
                std::array<int,3>{-1,-1,-1}, std::array<int,3>{1,-1,-1}, std::array<int,3>{1,1,-1}, std::array<int,3>{1,1,1},
                std::array<int,3>{-1,1,-1}, std::array<int,3>{-1,-1,1}, std::array<int,3>{-1,1,1}, std::array<int,3>{1,-1,1}}};
            for (auto const& [dirx,diry,dirz] : corners) startSendRcv(dirx,diry,dirz);
        }

        return countComm_;
    }

    template<int Mask>
    void communicateFillHaloStartAndWaitAll(T* data){
        int countMsg = communicateFillHaloStart<Mask>(data);
        //std::cout<< " After communicateFillHaloStart - n msg "<< countMsg <<std::endl;
        communicateWaitAllSendAndCheck(countMsg);
        //std::cout<< " After waitall send communication" <<std::endl;
        communicateWaitAllRcvAndCheck(countMsg);
        std::cout<< " End communication" <<std::endl;
    }


    template<int Mask>
    int copyBorderToHaloSelf(T* data){

        constexpr int communicationDim = 3;
        MPI_Comm  fieldComm  = MPIManager::getInstance().getFieldComm();
        const auto myCoords = MPIManager::getInstance().getCoordsField();
        const auto myrank = MPIManager::getInstance().getFieldRank();
        int countCopy = 0;
        auto copyBorderHalo = [&](int dirx,int diry,int dirz) -> void {

            const int idxDirectionComm = (dirx + 1) + (diry + 1) * communicationDim + (dirz + 1) * communicationDim * communicationDim;
            assert(idxDirectionComm != 13);
            // idxDirectionCommRcv = idxDirectionComm of the other rank that is sending data
            //mpi_check( MPI_Sendrecv(data, 1, border_type[idxDirectionComm], 0, 0, data, 1, halo_type[idxDirectionComm], 0, 0, MPI_COMM_SELF, &status) );
            mpi_check( MPI_Irecv(data, 1, halo_type[idxDirectionComm], myrank, idxDirectionComm, fieldComm, &requestsRcv_[countCopy]) );
            mpi_check( MPI_Isend(data, 1, border_type[idxDirectionComm], myrank, idxDirectionComm, fieldComm, &requestsSend_[countCopy]) );
            countCopy++;
        };
        // communicate faces
        if constexpr (Mask & FACES){
            constexpr std::array<std::array<int,3>,6> faces = {{ 
                std::array<int,3>{-1,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{0,-1,0}, 
                std::array<int,3>{0,1,0}, std::array<int,3>{0,0,-1}, std::array<int,3>{0,0,1} }};
            for (auto const& [dirx,diry,dirz] : faces) copyBorderHalo(dirx,diry,dirz);
        }

        // communicate edges
        if constexpr (Mask & EDGES){
            constexpr std::array<std::array<int,3>,12> edges = {{ 
                std::array<int,3>{-1,-1,0}, std::array<int,3>{1,-1,0}, std::array<int,3>{-1,1,0}, std::array<int,3>{1,1,0},
                std::array<int,3>{-1,0,-1}, std::array<int,3>{1,0,-1}, std::array<int,3>{-1,0,1}, std::array<int,3>{1,0,1},
                std::array<int,3>{0,-1,-1}, std::array<int,3>{0,1,-1}, std::array<int,3>{0,-1,1}, std::array<int,3>{0,1,1} }};
            for (auto const& [dirx,diry,dirz] : edges) copyBorderHalo(dirx,diry,dirz);
        }
        
        // communicate corners
        if constexpr (Mask & CORNERS){
            constexpr std::array<std::array<int,3>,8> corners = {{ 
                std::array<int,3>{-1,-1,-1}, std::array<int,3>{1,-1,-1}, std::array<int,3>{1,1,-1}, std::array<int,3>{1,1,1},
                std::array<int,3>{-1,1,-1}, std::array<int,3>{-1,-1,1}, std::array<int,3>{-1,1,1}, std::array<int,3>{1,-1,1}}};
            for (auto const& [dirx,diry,dirz] : corners) copyBorderHalo(dirx,diry,dirz);
        }

        communicateWaitAllSendAndCheck(countCopy);
        communicateWaitAllRcvAndCheck(countCopy);
        return countCopy;
    }

    template<int Mask>
    int copyHaloToBorderSelf(T* data){

        constexpr int communicationDim = 3;
        MPI_Comm  fieldComm  = MPIManager::getInstance().getFieldComm();
        const auto myCoords = MPIManager::getInstance().getCoordsField();
        const auto myrank = MPIManager::getInstance().getFieldRank();
        int countCopy = 0;
        
        auto copyHaloBorder = [&](int dirx,int diry,int dirz) -> void {
            std::cout<< "In copyHaloBorder lambda"  <<std::endl;
            const int idxDirectionComm = (dirx + 1) + (diry + 1) * communicationDim + (dirz + 1) * communicationDim * communicationDim;
            assert(idxDirectionComm != 13);
            // idxDirectionCommRcv = idxDirectionComm of the other rank that is sending data
            //MPI_Status status;
            //mpi_check( MPI_Sendrecv(data, 1, halo_type[idxDirectionComm], 0, 0, data, 1, border_type[idxDirectionComm], 0, 0, MPI_COMM_SELF, &status) );
            mpi_check( MPI_Irecv(data, 1, border_type[idxDirectionComm], myrank, idxDirectionComm, fieldComm, &requestsRcv_[countCopy]) );
            mpi_check( MPI_Isend(data, 1, halo_type[idxDirectionComm], myrank, idxDirectionComm, fieldComm, &requestsSend_[countCopy]) );
            countCopy++;
        };
        
        // communicate faces
        if constexpr (Mask & FACES){
            constexpr std::array<std::array<int,3>,6> faces = {{ 
                std::array<int,3>{-1,0,0}, std::array<int,3>{1,0,0}, std::array<int,3>{0,-1,0}, 
                std::array<int,3>{0,1,0}, std::array<int,3>{0,0,-1}, std::array<int,3>{0,0,1} }};
            for (auto const& [dirx,diry,dirz] : faces) copyHaloBorder(dirx,diry,dirz);
        }

        // communicate edges
        if constexpr (Mask & EDGES){
            constexpr std::array<std::array<int,3>,12> edges = {{ 
                std::array<int,3>{-1,-1,0}, std::array<int,3>{1,-1,0}, std::array<int,3>{-1,1,0}, std::array<int,3>{1,1,0},
                std::array<int,3>{-1,0,-1}, std::array<int,3>{1,0,-1}, std::array<int,3>{-1,0,1}, std::array<int,3>{1,0,1},
                std::array<int,3>{0,-1,-1}, std::array<int,3>{0,1,-1}, std::array<int,3>{0,-1,1}, std::array<int,3>{0,1,1} }};
            for (auto const& [dirx,diry,dirz] : edges) copyHaloBorder(dirx,diry,dirz);
        }
        
        // communicate corners
        if constexpr (Mask & CORNERS){
            constexpr std::array<std::array<int,3>,8> corners = {{ 
                std::array<int,3>{-1,-1,-1}, std::array<int,3>{1,-1,-1}, std::array<int,3>{1,1,-1}, std::array<int,3>{1,1,1},
                std::array<int,3>{-1,1,-1}, std::array<int,3>{-1,-1,1}, std::array<int,3>{-1,1,1}, std::array<int,3>{1,-1,1}}};
            for (auto const& [dirx,diry,dirz] : corners) copyHaloBorder(dirx,diry,dirz);
        }

        communicateWaitAllSendAndCheck(countCopy);
        communicateWaitAllRcvAndCheck(countCopy);

        return countCopy;
    }

private:
    uint halo[3];
    std::array<MPI_Datatype,27> border_type;
    std::array<MPI_Datatype,27> halo_type;
    std::array<int,27> neighbour_;
    int countComm_ = 0;
    MPI_Request* requestsSend_;
    MPI_Request* requestsRcv_;
    MPI_Status*  statusesSend_;
    MPI_Status*  statusesRcv_;
};