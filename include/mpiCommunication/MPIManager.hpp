#pragma once

#include <mpi.h>
#include <vector>
#include <assert.h>

#ifndef NO_MPI
#define NO_MPI 0
#endif

//using uint = std::uint32_t;



class MPIException : public std::runtime_error {
public:
    MPIException(int error_code) : std::runtime_error(build_msg(error_code)){}

private:
    static std::string build_msg(int error_code) {
        char buf[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(error_code, buf, &len);
        return std::string(buf, len);
    }
};

inline void mpi_check(int err) {
    if (err != MPI_SUCCESS) {
        throw MPIException(err);
    }
}


// Singleton class to manage MPI context and Cartesian communicator
class MPIManager {
public:
    // Access the singleton instance
    static MPIManager& getInstance() {
        assert(MPIManagerInitialized_);
        static MPIManager instance;
        return instance;
    }

    static void initGlobal(int *argc, char ***argv){
        assert(!MPIManagerInitialized_);
#if NO_MPI
        globalRank_ = 0;
        globalNprocs_ = 1;
#else
        mpi_check( MPI_Init(argc, argv) );
        mpi_check( MPI_Comm_dup(MPI_COMM_WORLD, &PIC_GLOBAL_COMM) );
        mpi_check( MPI_Comm_rank(PIC_GLOBAL_COMM, &globalRank_) );
        mpi_check( MPI_Comm_size(PIC_GLOBAL_COMM, &globalNprocs_) );
#endif
        MPIManagerInitialized_ = true;
        fieldCommInitialized_ = false;
        particleCommInitialized_ = false;
    }

    void initCartesianCommunicatorField(const int Dim, const int dimensions[], const bool periodic[], bool reorder=false) {
        assert(MPIManagerInitialized_);
#if NO_MPI
        fieldRank_ = globalRank_;
        fieldNprocs_ = globalNprocs_;
#else 
        dimField_ = Dim;
        std::vector<int> periodicInt(Dim);
        for(int i=0; i<dimField_; ++i) {
            periodicInt[i] = periodic[i] ? 1 : 0;
        }
        dimensionsField_.assign(dimensions, dimensions + dimField_);
        periodicField_.assign(periodic, periodic + dimField_);

        // Create Cartesian communicator
        mpi_check( MPI_Cart_create(PIC_GLOBAL_COMM, dimField_, dimensionsField_.data(), periodicInt.data(), reorder, &FIELD_COMM) );
        MPI_Comm_set_errhandler(FIELD_COMM, MPI_ERRORS_RETURN);
        mpi_check( MPI_Comm_size(FIELD_COMM, &fieldNprocs_) );
        mpi_check( MPI_Comm_rank(FIELD_COMM, &fieldRank_) );
        // Get this process's coordinates in the Cartesian grid
        coordinatesField_.assign(dimField_, 0);
        mpi_check( MPI_Cart_coords(FIELD_COMM, fieldRank_, dimField_, coordinatesField_.data()) );
#endif
        fieldCommInitialized_ = true;
    }

    void initCartesianCommunicatorParticles(const int Dim, const int dimensions[], const bool periodic[], bool reorder=false) {
        assert(MPIManagerInitialized_);
#if NO_MPI
        particleRank_ = globalRank_;
        particleNprocs_ = globalNprocs_;
#else
        dimParticle_ = Dim;
        std::vector<int> periodicInt(Dim);
        for(int i=0; i<dimParticle_; ++i) {
            periodicInt[i] = periodic[i] ? 1 : 0;
        }
        dimensionsParticle_.assign(dimensions, dimensions + dimParticle_);
        periodicParticle_.assign(periodic, periodic + dimParticle_);

        // Create Cartesian communicator
        mpi_check( MPI_Cart_create(PIC_GLOBAL_COMM, dimParticle_, dimensionsParticle_.data(), periodicInt.data(), reorder, &PARTICLE_COMM) );
        mpi_check( MPI_Comm_size(PARTICLE_COMM, &particleNprocs_) );
        mpi_check( MPI_Comm_rank(PARTICLE_COMM, &particleRank_) );
        // Get this process's coordinates in the Cartesian grid
        coordinatesParticle_.assign(dimParticle_, 0);
        mpi_check( MPI_Cart_coords(PARTICLE_COMM, particleRank_, dimParticle_, coordinatesParticle_.data()) );
#endif
        particleCommInitialized_ = true;
    }

    static void finalize_mpi() {
    #if NO_MPI
    #else
        // free particle communicator
        if(particleCommInitialized_){
            mpi_check( MPI_Comm_free(&PARTICLE_COMM) );
            particleCommInitialized_ = false;
        }
        // free field communicator
        if(fieldCommInitialized_){
            mpi_check( MPI_Comm_free(&FIELD_COMM) );
            fieldCommInitialized_ = false;
        }
        // free duplicated global communicator
        if(MPIManagerInitialized_){
            mpi_check( MPI_Comm_free(&PIC_GLOBAL_COMM) );
            MPIManagerInitialized_ = false;
        }
        // call MPI_Finalize
        mpi_check( MPI_Finalize() );
    #endif
    }

    // Query
    MPI_Comm getGlobalComm() const { return MPIManagerInitialized_ ? PIC_GLOBAL_COMM : MPI_COMM_NULL; }
    int getGlobalRank() const { return globalRank_; }
    int getGlobalNprocesses() const { return globalNprocs_; }
    
    MPI_Comm getFieldComm() const { return fieldCommInitialized_ ? FIELD_COMM : MPI_COMM_NULL; }
    int getFieldRank() const { return fieldRank_; }
    int getFieldNprocesses() const { return fieldNprocs_; }
    const std::vector<int>& getDimensionsField() const { return dimensionsField_; }
    const std::vector<bool>& getPeriodicField() const { return periodicField_; }
    const std::vector<int>& getCoordsField() const { return coordinatesField_; }
    
    MPI_Comm getParticleComm() const { return particleCommInitialized_ ? PARTICLE_COMM : MPI_COMM_NULL; }    
    int getParticleRank() const { return particleRank_; }
    int getParticleNprocesses() const { return particleNprocs_; }
    const std::vector<int>& getDimensionsParticle() const { return dimensionsParticle_; }
    const std::vector<bool>& getPeriodicParticle() const { return periodicParticle_; }
    const std::vector<int>& getCoordsParticle() const { return coordinatesParticle_; }
    
private:
    MPIManager(){}
    ~MPIManager(){}

    // disable copy and assignment
    MPIManager(const MPIManager&) = delete;
    MPIManager& operator=(const MPIManager&) = delete;

    static MPI_Comm PIC_GLOBAL_COMM;
    static MPI_Comm FIELD_COMM;
    static MPI_Comm PARTICLE_COMM;
    static int globalRank_, globalNprocs_;

    int dimField_;
    std::vector<int> dimensionsField_;
    std::vector<bool> periodicField_;
    int fieldRank_, fieldNprocs_;
    std::vector<int> coordinatesField_;
    
    int dimParticle_;
    std::vector<int> dimensionsParticle_;
    std::vector<bool> periodicParticle_;
    int particleRank_, particleNprocs_;
    std::vector<int> coordinatesParticle_;
    
    static bool MPIManagerInitialized_;
    static bool fieldCommInitialized_;
    static bool particleCommInitialized_;

};

MPI_Comm MPIManager::PIC_GLOBAL_COMM = MPI_COMM_NULL;
MPI_Comm MPIManager::FIELD_COMM = MPI_COMM_NULL;
MPI_Comm MPIManager::PARTICLE_COMM = MPI_COMM_NULL;

int MPIManager::globalRank_ = 0;
int MPIManager::globalNprocs_ = 0;

bool MPIManager::MPIManagerInitialized_ = false;
bool MPIManager::fieldCommInitialized_ = false;
bool MPIManager::particleCommInitialized_ = false;