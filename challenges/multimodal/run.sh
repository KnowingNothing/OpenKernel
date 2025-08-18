#!/bin/bash
#----------------------------------------------------------
# SBATCH Directives
#----------------------------------------------------------
#SBATCH --job-name=jacobi-2d_run
#SBATCH --output=./logs/run_mot_train_%j.out
#SBATCH --error=./logs/run_mot_train_%j.err
#SBATCH --partition=Long
#SBATCH --gres=gpu:4090:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 注意！运行脚本的时候不要load python环境，因此conda里面已经有安装，再次安装会导致环境冲突
echo "================================================================"
echo "作业ID: $SLURM_JOB_ID, 运行节点: $(hostname), 开始时间: $(date)"
echo "================================================================"

# 1. 【新增】加载Spack管理的CUDA环境
#    spack load 命令会自动设置 PATH 和 CUDA_HOME 等环境变量
echo "正在加载 Spack CUDA 模块..."
spack load cuda@12.4.1

# 2. 激活您的 Conda 环境
echo "正在激活 Conda 环境: bagel..."
# conda activate bagel
source ~/miniconda3/etc/profile.d/conda.sh  # 根据你的实际路径改
conda activate bagel

# export PYTORCH_SDP_BACKEND=math


# 3. 检查环境，并打印关键环境变量（用于调试）
echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME 环境变量: $CUDA_HOME"
echo "nvcc 路径: $(which nvcc)"
echo "Python 路径: $(which python)"
echo "----------------------------------------------------------------"

# python run.py
# cd /home/zhwh/OpenKernel/challenges/multimodal
cd /home/zhwh/OpenKernel/challenges/multimodal/custom_kernel
pip uninstall custom_kernel -y
rm -rf build dist *.egg-info
find . -name "*.so" -delete
pip install -e . --no-cache-dir

cd /home/zhwh/OpenKernel/challenges/multimodal
python verify.py
# python run.py

# pip install -e .
# python test_mask.py

echo "================================================================"
echo "作业结束时间: $(date)"
echo "================================================================"