from .._typing import Array as A
from ..base import function
from ..jit.placeholder import FT_Tracer
from ..jit.utils import name
from ...backend.backend import xp
from .primitive_reduct import sum

def broadcast_backward(grad: A, x_shape: tuple):
    # Remove leading dims added by broadcasting
    while len(grad.shape) > len(x_shape):
        grad = sum(grad, axis=0)

    # Reduce along broadcasted axes
    for i, (sx, sg) in enumerate(zip(x_shape, grad.shape)):
        if sx == 1 and sg != 1:
            grad = sum(grad, axis=i, keepdims=True)

    return grad
