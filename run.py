
import faketensor as ft 
from faketensor import ndarray as nd
import numpy as np

np.random.seed(0)
a = nd.array(np.random.rand(4, 5))
b = nd.array(np.random.rand(5, 4))

def fn(a):
    return a.T

print(a)

print(ft.value_and_grad(fn)([a]))

