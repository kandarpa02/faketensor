from faketensor.src.jit.placeholder import FT_Tracer
from faketensor.src.jit.executor import FT_Function
import faketensor as ft 
from faketensor import ndarray as nd
from faketensor import functions as f

a = nd.array(4.)
b = nd.array(3.)

# @ft.jit.trace
def fn(a, b):
    return (lambda x, y:(x * y)+x)(a, b)

print(fn(a, b))

print(ft.jit.trace(ft.grad(lambda x, y:(x * y)+x))(a, b))


# a = FT_Tracer((), 'float32', 'a')
# b = FT_Tracer((), 'float32', 'b')
# c = FT_Tracer((), 'float32', 'c')
# out = (a * b)+a

# f = FT_Function(out, [a, b])

# print(a, b)
# e = f.compile()

# print('out:\n',e(nd.array(4.), nd.array(3.)))

import numpy as np
