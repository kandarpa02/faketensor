from .._typing import Array as A
from ..base import function
from ..utils import broadcast_backward
from ..jit.placeholder import FT_Tracer
from ..jit.utils import name
from ...backend.backend import xp

# Generic type
Array = A | int | float


# ============================================================
# ADD
# ============================================================

def add(x: Array, y: Array):
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, y.shape)
            return g1, g2

        out = as_nd(lib.add(x, y))
        return out, (as_nd(x), as_nd(y)), grad_fn

    def static_fun(x: FT_Tracer, y: FT_Tracer):
        return x + y

    return function(_fun, static_fun)(x, y)


# ============================================================
# SUBTRACT
# ============================================================

def subtract(x: Array, y: Array):
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(negative(g), y.shape)
            return g1, g2

        out = as_nd(lib.subtract(x, y))
        return out, (as_nd(x), as_nd(y)), grad_fn

    def static_fun(x: FT_Tracer, y: FT_Tracer):
        return x - y

    return function(_fun, static_fun)(x, y)


# ============================================================
# NEGATIVE
# ============================================================

def negative(x: Array):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(negative(g), x.shape)
            return g1,

        out = as_nd(lib.negative(x))
        return out, (as_nd(x),), grad_fn

    def static_fun(x: FT_Tracer):
        return -x

    return function(_fun, static_fun)(x)


# ============================================================
# MULTIPLY
# ============================================================

def multiply(x: Array, y: Array):
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.multiply(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(multiply(g, y), x.shape)
            g2 = broadcast_backward(multiply(g, x), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    def static_fun(x: FT_Tracer, y: FT_Tracer):
        return x * y

    return function(_fun, static_fun)(x, y)


# ============================================================
# DIVIDE
# ============================================================

def divide(x: Array, y: Array):
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.divide(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(multiply(g, divide(as_nd(1.0), y)), x.shape)
            g2 = broadcast_backward(
                negative(multiply(g, multiply(x, power(y, as_nd(-2))))),
                y.shape
            )
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    def static_fun(x: FT_Tracer, y: FT_Tracer):
        return x / y

    return function(_fun, static_fun)(x, y)


# ============================================================
# LOG
# ============================================================

def log(x: Array):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        out = as_nd(lib.log(x))

        def grad_fn(g):
            return multiply(g, divide(as_nd(1.0), x)),

        return out, (as_nd(x),), grad_fn

    def static_fun(x: FT_Tracer):
        return FT_Tracer(x.shape, x.dtype, name)

    return function(_fun, static_fun)(x)


# ============================================================
# POWER
# ============================================================

def power(x: Array, y: Array):
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.power(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(
                multiply(g, multiply(y, power(x, subtract(y, as_nd(1))))) ,
                x.shape
            )
            g2 = broadcast_backward(
                multiply(g, multiply(out, log(x))),
                y.shape
            )
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    def static_fun(x: FT_Tracer, y: FT_Tracer):
        return x ** y

    return function(_fun, static_fun)(x, y)
