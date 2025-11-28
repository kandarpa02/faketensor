from .executor import *
from typing import NamedTuple, Callable

class StaticFunction(NamedTuple):
    fun:Callable

    def __call__(self, *args):
        output = self.fun(*args)
        return output
    
    @staticmethod
    def get_arguments(out):
        def topo_sort(node):
            order = []
            visited = set()

            def dfs(n):
                if id(n) in visited:
                    return
                visited.add(id(n))

                if not hasattr(n, "parents"):
                    order.append(n)
                    return

                for p in n.parents:
                    dfs(p)

                order.append(n)

            dfs(node)
            return order

        def get_ordered_tracer_inputs(out):
            nodes = topo_sort(out)
            seen = set()
            ordered_inputs = []

            for n in nodes:
                if not hasattr(n, "parents"):
                    continue
                for p in n.parents:
                    if isinstance(p, FT_Tracer) and p.is_leaf():
                        if id(p) not in seen:
                            seen.add(id(p))
                            ordered_inputs.append(p)

            return ordered_inputs
        
        return get_ordered_tracer_inputs(out)

class FunctionContext(NamedTuple):
    func:Callable
    def __call__(self, *args):
        return self.func(*args)
    

def trace(fun):
    def wrapper(*args):
        import faketensor.src.jit.placeholder as p

        prev = p.TRACING
        p.TRACING = True
        try:
            out = StaticFunction(fun)(*args)
            arguments = StaticFunction.get_arguments(out)
            print("Args\n", arguments)
            f = FT_Function(out=out, variables=arguments)
            # print(f)
            run = FunctionContext(f.compile())
        finally:
            p.TRACING = prev
        # print(args)
        return run(*args)

    return wrapper

