#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name=initial_chain
#SBATCH --mail-user=oliviarose004@protonmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --time=010:00:00
#SBATCH --account=oz059

module purge
ml gcc/12.2.0
ml openmpi/4.1.4
source activate NGC5102

mpirun -n 20  python JAM_MCMC.py

