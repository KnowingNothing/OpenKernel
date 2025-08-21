#!/bin/bash
#SBATCH --job-name=HPCGame
#SBATCH --output=./logs/run_mot_train_%j.out
#SBATCH --error=./logs/run_mot_train_%j.err
#SBATCH --partition=Long
#SBATCH --gres=gpu:4090:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Note! Do not load a python environment when running the script, because one is already installed in conda.
# Loading another one will cause environment conflicts.
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID, Running on node: $(hostname), Start Time: $(date)"
echo "================================================================"

echo "Loading Spack CUDA module..."
spack load cuda@12.4.1

echo "Activating Conda environment: bagel..."
# Activate conda
## Replace the path below with the path to your own conda script
source ~/miniconda3/etc/profile.d/conda.sh  # Change according to your actual path
conda activate bagel

echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME Environment Variable: $CUDA_HOME"
echo "nvcc Path: $(which nvcc)"
echo "Python Path: $(which python)"
echo "----------------------------------------------------------------"

# # Run the actual script
# python verify.py

# install the package
pip install -e .

echo "================================================================"
echo "Job End Time: $(date)"
echo "================================================================"