#!/bin/bash
#----------------------------------------------------------
# SBATCH Directives
#----------------------------------------------------------
#SBATCH --job-name=run_bagel
#SBATCH --output=./logs/run_bagel_%j.out
#SBATCH --error=./logs/run_bagel_%j.err
#SBATCH --gres=gpu:4090:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

# Note! Do not load a python environment when running the script, as one is already installed in conda. Loading another will cause environment conflicts.
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID, Running on node: $(hostname), Start Time: $(date)"
echo "================================================================"

# The spack load command will automatically set environment variables like PATH and CUDA_HOME
echo "Loading CUDA module..."
# spack load cuda@12.4.1
module load cuda/12.4

# Activate Conda environment
echo "Activating Conda environment: bagel..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate bagel

# Print environment variables (for debugging)
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME Environment Variable: $CUDA_HOME"
echo "nvcc Path: $(which nvcc)"
echo "Python Path: $(which python)"
echo "----------------------------------------------------------------"


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# python verify.py
python run.py

echo "================================================================"
echo "Job End Time: $(date)"
echo "================================================================"
