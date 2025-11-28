from .._typing import Array as A
from ..base import function
from ..utils import broadcast_backward
from ..jit.placeholder import FT_Tracer
from ..jit.utils import name
from ...backend.backend import xp
from .primitive_array_ops import reshape
# Generic type
Array = A


def sum(x: Array, axis=None, keepdims=False):
    lib = xp()
    shape_ = ()

    def _fun(x):
        from ..array import as_nd

        out = as_nd(lib.sum(x, axis, keepdims=keepdims))
        global shape_
        shape_ = out.shape
        def grad_fn(g):
            return as_nd(lib.full(x.shape, g)),
        
        return out, (as_nd(x),), grad_fn
    
    def static_fun(x):
        return FT_Tracer(shape_, x.dtype, 'sum', lambda x: lib.sum(x, axis=axis, keepdims=keepdims))
    
    return function(_fun, static_fun)(x)


