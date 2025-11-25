from dataclasses import dataclass
from typing import Callable, Any
from contextlib import contextmanager
from .utils import Meta
from ...backend import backend as b

TRACING = False
JIT_STACK = []

@contextmanager
def trace_mode():
    JIT_STACK.append([])  
    try:
        yield
    finally:
        pass   

def element_wise(a, b, name, func):
    shape = Meta.element_wise_shape(a, b)
    dtype = Meta.DType(a, b)
    out = FT_Tracer(
        shape,
        dtype,
        # device=a.dtype,
        name=name,
        func=func,
        parents=(a, b)
    )
    return out

def vectorize(a, b, name, func):
    shape = Meta.dot_shape(a, b)
    dtype = Meta.DType(a, b)
    out = FT_Tracer(
        shape,
        dtype,
        # device=a.dtype,
        name=name,
        func=func,
        parents=(a, b)
    )
    return out

class FT_Tracer:
    def __init__(self, shape:tuple, dtype:str, name:str='', func:Callable=lambda:None, parents:tuple=(), device='auto') -> None:
        self.shape = shape
        self.dtype = dtype
        self.lib = b.get_device()
        self.name = name
        self.func = func
        self.parents = parents

    def is_leaf(self):
        return len(self.parents) == 0

    def __repr__(self):
        return f"FT_Tracer(shape={self.shape}, dtype='{self.dtype}', name='{self.name}')"
    
    def __str__(self): return self.__repr__()

    def __add__(self, other):
        return element_wise(self, other, 'add', lambda a, b: a + b)
    
    def __mul__(self, other):
        return element_wise(self, other, 'mul', lambda a, b: a * b)

    def __sub__(self, other):
        return element_wise(self, other, 'sub', lambda a, b: a - b)
    
    def __truediv__(self, other):
        return element_wise(self, other, 'div', lambda a, b: a / b)
    
    def __neg__(self):
        return FT_Tracer(self.shape, self.dtype, 'neg', lambda a: -a)
    
    def __pow__(self, other):
        return FT_Tracer(self.shape, self.dtype, 'pow', lambda a, b: a ** b)
    
    @staticmethod
    def F_log(a):
        lib = a.lib
        return FT_Tracer(a.shape, a.dtype, 'log', lambda a: lib.log(a))
    
    @staticmethod
    def F_log10(a):
        lib = a.lib
        return FT_Tracer(a.shape, a.dtype, 'log10', lambda a: lib.log10(a))
    
    # def __matmul__(self, other):
    #     return vectorize(self, other, 'matmul', )

