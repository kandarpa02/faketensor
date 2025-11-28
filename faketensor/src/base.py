from typing import List, Callable, Protocol, Union
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from .utils import broadcast_backward
from .jit.placeholder import FT_Tracer, trace_mode
from .jit.utils import next_name
from ._typing import Array 

_RECORDING = True

TAPE_STACK = []

def active_tape():
    return TAPE_STACK[-1] if TAPE_STACK else None

@contextmanager
def tape():
    TAPE_STACK.append([])  
    try:
        yield
    finally:
        pass   

@contextmanager
def no_record():
    global _RECORDING
    prev = _RECORDING
    _RECORDING = False
    try:
        yield
    finally:
        _RECORDING = prev

@dataclass
class Node:
    out: Array
    parents: tuple
    grad_fn: Callable


class function:
    def __init__(self, fun, static_fun):
        self.fun = fun
        self.static_fun = static_fun

    def __call__(self, *args):

        next_name()
        def assign(arg):
            from .jit.utils import name
            next_name()
            return FT_Tracer(arg.shape, arg.dtype.__str__() , name)
        
        static_args = [assign(arg) for arg in args]

        static_output = self.static_fun(*static_args)

        global _RECORDING
        prev = _RECORDING
        _RECORDING = False

        try:
            output = self.fun(*args)

            if not isinstance(output, tuple):
                raise TypeError(
                    f"Function '{self.fun.__name__}' must return a tuple"
                )

            n = len(output)

            if n == 3:
                out, parents, grad_fn = output
                if not isinstance(parents, (tuple, list)):
                    raise TypeError("parents must be tuple/list")

            elif n == 2:
                out, grad_fn = output
                parents = args

            else:
                raise ValueError("Function must return (out, parents, grad_fn) or (out, grad_fn)")
            
            if not callable(grad_fn):
                raise TypeError("grad_fn must be callable.")

        finally:
            _RECORDING = prev

        # append to tape for dynamic mode
        t = active_tape()
        if t is not None and _RECORDING:
            t.append(Node(out, parents, grad_fn))
        
        from .jit.placeholder import TRACING

        return out if not TRACING else static_output
