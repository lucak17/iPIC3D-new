#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --time=00:00:50
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output test.o
#SBATCH --error test.e


srun ./testCommunicatorField 2 2 4