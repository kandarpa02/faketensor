from .._typing import Array as A
from ..base import function
from ..utils import broadcast_backward
from ...backend.backend import xp

# Generic type
Array = A


def reshape(x: Array, shape: tuple|list):
    lib = xp()

    prev_shape = x.shape

    def _fun(x):
        from ..array import as_nd

        out = as_nd(lib.reshape(x, shape))

        def grad_fn(g):
            g0 = reshape(g, prev_shape)
            return g0,
        
        return out, (as_nd(x),), grad_fn
    
    return function(_fun)(x)
