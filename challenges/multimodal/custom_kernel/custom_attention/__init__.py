# my_custom_attention/__init__.py
from . import _C

# 将 C++ 模块中的 'forward' 函数导入到包的顶层
from ._C import forward

__all__ = ['forward']