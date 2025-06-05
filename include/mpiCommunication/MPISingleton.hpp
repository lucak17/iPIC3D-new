#pragma once

#include <mpi.h>
#include <vector>
#include <assert.h>

#define NO_MPI = 0

using uint = std::uint32_t;

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
        this->globalRank_ = 0;
        this->globalNprocs_ = 1;
        this->fieldRank = this->globalRank_;
        this->fieldNprocs_ = this->globalNprocs__;
        this->particleRank = this->globalRank_;
        this->particleNprocs_ = this->globalNprocs__;
#else
        MPI_Init(argc, argv);
        MPI_Comm_dup(MPI_COMM_WORLD, &PIC_GLOBAL_COMM);
        MPI_Comm_rank(PIC_GLOBAL_COMM, &globalRank_);
        MPI_Comm_size(PIC_GLOBAL_COMM, &globalNprocs_);
#endif    
        MPIManagerInitialized_ = true;
    }

    static void initCartesianCommunicatorField(const uint Dim, const uint dimensions[], const bool periodic[], bool reorder=false) {
        assert(MPIManagerInitialized_);
#ifndef NO_MPI
        this->dimField_ = Dim;
        this->dimensionsField_.assign(dimensions, dimensions + dimField_);
        this->periodicField_.assign(periodic, periodic + dimField_);

        // Create Cartesian communicator
        MPI_Cart_create(PIC_GLOBAL_COMM, dimField_,dimensionsField_,reinterpret_cast<const int*>(periodicField_),reorder, &FIELD_COMM);
        MPI_Comm_size(MPI_COMM_WORLD, &fieldNprocs_);
        MPI_Comm_rank(MPI_COMM_WORLD, &fieldRank_);
        // Get this process's coordinates in the Cartesian grid
        coords_.assign(dimField_, 0);
        MPI_Cart_coords(FIELD_COMM, fieldRank_, dimField_, coords_.data());
#endif
        fieldCommInitialized_ = true;
    }

    static void initCartesianCommunicatorParticles(const uint Dim, const uint dimensions[], const bool periodic[], bool reorder=false) {
        assert(MPIManagerInitialized_);
#ifndef NO_MPI
        this->dimParticle_ = Dim;
        this->dimensionsParticles_.assign(dimensions, dimensions + dimParticles_);
        this->periodicParticle_.assign(periodic, periodic + dimParticles_);

        // Create Cartesian communicator
        MPI_Cart_create(PIC_GLOBAL_COMM, dimParticle_, dimensionsParticle_, reinterpret_cast<const int*>(periodicParticles_), reorder, &PARTICLE_COMM);
        MPI_Comm_size(MPI_COMM_WORLD, &particleNprocs_);
        MPI_Comm_rank(MPI_COMM_WORLD, &particleRank_);
        // Get this process's coordinates in the Cartesian grid
        coords_.assign(dimParticles_, 0);
        MPI_Cart_coords(PARTICLE_COMM, fieldRank_, dimParticle_, coords_.data());
#endif
        particleCommInitialized_ = true;
    }

    static void finalize_mpi() {
    #ifndef NO_MPI
        MPI_Finalize();
    #endif
    }

    // Query
    MPI_Comm getGlobalComm() const { return MPIManagerInitialized_ ? PIC_GLOBAL_COMM : MPI_COMM_NULL; }
    uint getGlobalRank() const { return globalRank_; }
    uint getGlobalNprocesses() const { return globalNprocs_; }
    
    MPI_Comm getFieldComm() const { return fieldCommInitialized_ ? FIELD_COMM : MPI_COMM_NULL; }
    uint getFieldRank() const { return fieldRank_; }
    uint getFieldNprocesses() const { return fieldNprocs_; }
    const std::vector<unt>& getDimensionsField() const { return dimensionsField_; }
    const std::vector<bool>& getPeriodicField() const { return periodicField_; }
    const std::vector<int>& getCoordsField() const { return coordinatesField_; }
    
    MPI_Comm getParticleComm() const { return particleCommInitialized_ ? PARTICLE_COMM : MPI_COMM_NULL; }    
    uint getParticleRank() const { return particleRank_; }
    uint getParticleNprocesses() const { return particleNprocs_; }
    const std::vector<unt>& getDimensionsParticle() const { return dimensionsParticle_; }
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
    uint globalRank_, globalNprocs_;

    uint dimField_;
    std::vector<uint> dimensionsField_;
    std::vector<bool> periodicField_;
    uint fieldRank_, fieldNprocs_;
    std::vector<int> coordinatesField_;
    
    uint dimParticle_;
    std::vector<uint> dimensionsParticle_;
    std::vector<bool> periodicParticle_;
    uint particleRank_, particleNprocs_;
    std::vector<int> coordinatesParticle_;
    
    bool MPIManagerInitialized_ = false;
    bool fieldCommInitialized_ = false;
    bool particleCommInitialized_ = false;

};
