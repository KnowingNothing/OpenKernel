# Import necessary functions and classes
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    # Defines the distribution name of the python package when published
    name='custom_attention', # Package name
    # "Extension modules", indicating non-Python code that needs to be compiled
    ext_modules=[
        # Represents a CUDA extension module
        CUDAExtension(
            # Defines the full import path in Python for the shared library generated after compilation
            name='custom_attention._C', 
            # List of source files, indicating all code that needs to be compiled
            sources=[
                'custom_attention/binding.cpp',
                'custom_attention/custom_attention.cu',
            ]
        ),
    ],
    # Specify the class for the 'build_ext' command as BuildExtension
    cmdclass={
        'build_ext': BuildExtension
    },
    # Folders/importable packages containing python code (with an __init__.py file)
    packages=['custom_attention'] 
)