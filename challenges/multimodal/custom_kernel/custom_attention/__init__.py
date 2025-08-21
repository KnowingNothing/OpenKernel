# my_custom_attention/__init__.py
from . import _C

# Import the 'forward' function in the C++ module to the top layer of the package
from ._C import forward

__all__ = ['forward']