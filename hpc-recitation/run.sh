#!/bin/bash

#SBATCH --job-name=Comphys
#SBATCH --output=lorenz-%j.out
#SBATCH --error=lorenz-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=1G

module purge
module load anaconda3/2024.02
echo "Loaded modules"
srun --unbuffered python lorenz.py
echo "We finished!"