from .placeholder import FT_Tracer
from typing import NamedTuple, List, Callable


# ============================================================
# Topological sort (ONLY FT_Tracer nodes)
# ============================================================

def topo_sort(node):
    order = []
    visited = set()

    def dfs(n):
        if id(n) in visited:
            return
        visited.add(id(n))

        # skip non-tracers entirely
        if not isinstance(n, FT_Tracer):
            return

        for p in n.parents:
            dfs(p)

        order.append(n)

    dfs(node)
    return order


# ============================================================
# Instruction
# ============================================================

class Instruction(NamedTuple):
    func: Callable
    parent_ids: List[int]
    out_id: int


# ============================================================
# CompiledFunction
# ============================================================

class CompiledFunction:
    def __init__(self, out_index, instrs, var_indices, num_slots):
        self.out_index = out_index
        self.instrs = instrs
        self.var_indices = var_indices
        self.num_slots = num_slots

    def __call__(self, *args):
        # Buffer holds intermediate real NDArray values (not tracers)
        buf = [None] * self.num_slots

        # 1) Assign input args to their buffer slots
        for idx, val in zip(self.var_indices, args):
            buf[idx] = val

        # 2) Execute instructions in topological order
        for instr in self.instrs:
            func = instr.func
            pids = instr.parent_ids

            # Gather arguments for this node
            real_args = [buf[p] for p in pids]

            # Execute the primitive (real ND ops)
            buf[instr.out_id] = func(*real_args)

        # 3) Return output
        return buf[self.out_index]


# ============================================================
# FT_Function
# ============================================================

class FT_Function(NamedTuple):
    out: FT_Tracer
    variables: List[FT_Tracer]

    def compile(self) -> CompiledFunction:
        # 1) Topo sort of FT_Tracer graph
        nodes = topo_sort(self.out)

        # 2) Assign slots only to tracer nodes
        index = {id(n): i for i, n in enumerate(nodes)}
        print("IDX\n", index)

        # variable indices in buf[]
        var_indices = [index[id(v)] for v in self.variables]

        instrs = []

        # 3) Build IR instructions
        for n in nodes:
            # variables have no func
            if n in self.variables:
                continue

            # Parents that are FT_Tracer

            pids = [index.get(id(p), 0) for p in n.parents]
                
            print('PID\n', pids)

            instrs.append(Instruction(n.func, pids, index[id(n)]))

        out_index = index[id(self.out)]
        num_slots = len(nodes)

        return CompiledFunction(out_index, instrs, var_indices, num_slots)
