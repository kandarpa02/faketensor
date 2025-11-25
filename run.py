from faketensor.src.jit.placeholder import FT_Tracer
from faketensor.src.jit.executor import FT_Function
import faketensor as ft 
from faketensor import ndarray as nd

# a = FT_Tracer((), 'float32', 'a')
# b = FT_Tracer((), 'float32', 'b')
# c = FT_Tracer((), 'float32', 'c')
# out = (a * b)+a

# f = FT_Function(out, [a, b])

# print('func:\n', f)
# e = f.compile()

# print('out:\n',e(2., 3.))

a = nd.array(5.)
g = ft.grad(lambda x, y=4:x**y)
g2 = ft.grad(g)
g3 = ft.grad(g2)

print(a**4)
print('dx ', g(a))
print('d2x ', g2(a))
print('d3x ', g3(a))