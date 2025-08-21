#!/bin/bash
#----------------------------------------------------------
# SBATCH Directives
#----------------------------------------------------------
#SBATCH --job-name=HPCGame
#SBATCH --output=./logs/run_mot_train_%j.out
#SBATCH --error=./logs/run_mot_train_%j.err
#SBATCH --partition=Long
#SBATCH --gres=gpu:4090:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Note! Do not load a python environment when running the script, as one is already installed in conda. Loading another will cause environment conflicts.
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID, Running on node: $(hostname), Start Time: $(date)"
echo "================================================================"

# The spack load command will automatically set environment variables like PATH and CUDA_HOME
echo "Loading Spack CUDA module..."
spack load cuda@12.4.1

# Activate Conda environment
echo "Activating Conda environment: bagel..."
# ===== Note! The path below needs to be replaced with your own miniconda3 activation script path =====
source ~/miniconda3/etc/profile.d/conda.sh  # Modify according to your actual path
conda activate bagel

# Print environment variables (for debugging)
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME Environment Variable: $CUDA_HOME"
echo "nvcc Path: $(which nvcc)"
echo "Python Path: $(which python)"
echo "----------------------------------------------------------------"

# python verify.py
python run.py

echo "================================================================"
echo "Job End Time: $(date)"
echo "================================================================"