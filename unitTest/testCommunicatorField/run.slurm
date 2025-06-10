#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --time=00:00:30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output test.o
#SBATCH --error test.e


srun ./testCommunicatorField 1 1 1