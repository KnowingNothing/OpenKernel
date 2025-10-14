#!/bin/bash
#SBATCH --job-name=compile_custom_op
#SBATCH --output=./logs/compile_custom_op_%j.out
#SBATCH --error=./logs/compile_custom_op_%j.err
#SBATCH --gres=gpu:4090:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1

# Note! Do not load a python environment when running the script, because one is already installed in conda.
# Loading another one will cause environment conflicts.
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID, Running on node: $(hostname), Start Time: $(date)"
echo "================================================================"

echo "Loading CUDA module..."
module load cuda/12.4
# spack load cuda@12.4.1 
echo "Activating Conda environment: bagel..."
# Activate conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate bagel
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME Environment Variable: $CUDA_HOME"
echo "nvcc Path: $(which nvcc)"
echo "Python Path: $(which python)"
echo "----------------------------------------------------------------"


# 设置 PyTorch 库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# 检查 PyTorch 库路径
echo "检查 PyTorch 库路径..."
find $CONDA_PREFIX -name "libc10.so" 2>/dev/null

export TORCH_CUDA_ARCH_LIST="8.9"

echo "----------------------------------------------------------------"
echo "开始编译CUDA扩展..."

# 清理之前的编译
rm -rf build custom_attention/_C*.so

# 直接使用setup.py编译（避免pip的网络请求）
# cd /data/run01/scze029/OpenKernel/challenges/multimodal/custom_kernel
python setup.py build_ext --inplace

echo "检查编译结果..."
find . -name "*.so" -o -name "_C.*.so"

echo "验证安装..."
python -c "
try:
        import custom_attention
        print('✓ custom_attention 导入成功')
        import custom_attention._C
        print('✓ CUDA扩展导入成功')
except Exception as e:
        print('✗ 导入失败:', e)
        import traceback
        traceback.print_exc()
"

# 找到 conda 环境的 site-packages 路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages 路径: $SITE_PACKAGES"

# 创建目标目录
mkdir -p $SITE_PACKAGES/custom_attention

# 复制 Python 文件
cp custom_attention/__init__.py $SITE_PACKAGES/custom_attention/

# 复制编译好的 .so 文件
# 查找所有可能的 .so 文件位置
find . -name "_C*.so" -exec cp {} $SITE_PACKAGES/custom_attention/ \;

# 如果上面的命令没找到，尝试这些可能的位置
cp build/lib.linux-x86_64-cpython-310/custom_attention/_C*.so $SITE_PACKAGES/custom_attention/ 2>/dev/null || true
cp custom_attention/_C*.so $SITE_PACKAGES/custom_attention/ 2>/dev/null || true

# 检查复制结果
echo "复制后的文件:"
ls -la $SITE_PACKAGES/custom_attention/

# 检验是否可以在任意目录下使用
cd ..
python -c "
try:
        import custom_attention
        print('✓ custom_attention 导入成功')
        import custom_attention._C
        print('✓ CUDA 扩展导入成功')
except Exception as e:
        print('✗ 在其他目录下导入失败:', e)
"
cd custom_kernel

python verify.py
echo "================================================================"
echo "Job End Time: $(date)"
echo "================================================================"
