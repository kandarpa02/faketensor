from typing import Callable, Any, Tuple, Union
from ...backend import backend as b
from ..base import TAPE_STACK, tape
from ..array import NDarray


def _extract_np(x):
    """Return underlying numpy array for NDarray or raw numpy."""
    if isinstance(x, NDarray):
        return x.np
    return x


def _id(x):
    """ID based on numpy buffer for NDarray."""
    if isinstance(x, NDarray):
        return id(x.np)
    return id(x)


def _zero_like(x):
    return b.xp().zeros_like(_extract_np(x))


# ================================================================
# BACKWARD CORE (internal)
# ================================================================
def _backward(fun: Callable, args, argnums: Union[int, tuple, None]):
    """
    Perform forward + backward and return (value, grad_dict).
    grad_dict maps id(x) → numpy gradient.
    """

    if any(not isinstance(a, NDarray) for a in args):
        raise TypeError("Only NDarray arguments supported")
    
    # -----------------------------
    # 1) Forward pass (build tape)
    # -----------------------------
    with tape():
        output = fun(*args)

    tape_records = TAPE_STACK[-1] if TAPE_STACK else []

    # Init gradient for output
    grads = { _id(_extract_np(output)) : b.xp().ones_like(_extract_np(output)) }

    # -----------------------------
    # 2) Backward pass
    # -----------------------------
    for node in reversed(tape_records):
        g_out = grads.get(_id(_extract_np(node.out)))
        if g_out is None:
            continue

        parent_grads = node.grad_fn(g_out)

        for parent, parent_grad in zip(node.parents, parent_grads):
            pid = _id(_extract_np(parent))

            # normalize to numpy
            pg = _extract_np(parent_grad)

            if pid in grads:
                grads[pid] = NDarray(grads[pid] + pg)
            else:
                grads[pid] = NDarray(pg)

    return output, grads


# ================================================================
# PUBLIC API: grad()
# ================================================================
def grad(fun: Callable, argnum: Union[int, tuple, list, None] = None) -> Callable:
    """
    JAX-like grad supporting:
        - int -> single argument
        - tuple/list -> multiple arguments
        - None -> all arguments
    Accepts *args as unpacked or single list/tuple of NDarrays.
    """
    def wrapped(*args):
        # Flatten if user passed a single list/tuple
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args_flat = tuple(args[0])
        else:
            args_flat = args

        if any(not isinstance(a, NDarray) for a in args_flat):
            raise TypeError("Only NDarray arguments supported")

        out, gdict = _backward(fun, args_flat, argnums=argnum)

        # Normalize argnums
        if argnum is None:
            target_ids = [_id(a) for a in args_flat]
            indices = list(range(len(args_flat)))
        elif isinstance(argnum, int):
            target_ids = [_id(args_flat[argnum])]
            indices = [argnum]
        else:  # tuple or list
            target_ids = [_id(args_flat[i]) for i in argnum]
            indices = list(argnum)

        # Collect gradients
        results = [
            gdict.get(tid, _zero_like(args_flat[i]))
            for i, tid in zip(indices, target_ids)
        ]

        if isinstance(argnum, int):
            return results[0]
        res = tuple(results)
        return res[0] if len(res) == 1 else list(res)

    return wrapped



# ================================================================
# PUBLIC API: value_and_grad()
# ================================================================

def value_and_grad(fun: Callable, argnum: Union[int, tuple, list, None]=None) -> Callable:
    """
    Return (value, grads)
    Supports:
      - *args as unpacked NDarrays
      - single list/tuple of NDarrays
    """
    def wrapped(*args):
        # Flatten if user passed a single list/tuple
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args_flat = tuple(args[0])
        else:
            args_flat = args

        if any(not isinstance(a, NDarray) for a in args_flat):
            raise TypeError("Only NDarray arguments supported")

        out, gdict = _backward(fun, args_flat, argnums=argnum)

        # Normalize argnums
        if argnum is None:
            target_ids = [_id(a) for a in args_flat]
            grad_vals = tuple(gdict.get(t, _zero_like(a)) for t, a in zip(target_ids, args_flat))
        elif isinstance(argnum, int):
            target_id = _id(args_flat[argnum])
            grad_vals = gdict.get(target_id, _zero_like(args_flat[argnum]))
        else:  # tuple or list
            target_ids = [_id(args_flat[i]) for i in argnum]
            grad_vals = tuple(gdict.get(t, _zero_like(args_flat[i])) for t, i in zip(target_ids, argnum))

        return out, grad_vals[0] if isinstance(grad_vals, tuple) and len(grad_vals)==1 else list(grad_vals)

    return wrapped
