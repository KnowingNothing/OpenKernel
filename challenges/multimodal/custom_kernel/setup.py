# 这个文件应该放在自制kernel外面
# 导入必要的函数和类
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    # 定义了py包在发布时的分发名称
    name='custom_attention', # 包名
    # “扩展模块”，表示需要编译的非python代码
    ext_modules=[
        # 表示CUDA扩展模块
        CUDAExtension(
            # 定义了编译完成后生成的共享库在 Python 中的完整导入路径
            name='custom_attention._C', 
            # 源文件列表，表示所有需要被编译的代码
            sources=[
                'custom_attention/binding.cpp',
                'custom_attention/custom_attention.cu',
            ]
        ),
    ],
    # 指定构建build_ext的方式为BuildExtension类
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['custom_attention'] # 包含python代码的文件夹/可导入包（含有__init__.py文件）
)