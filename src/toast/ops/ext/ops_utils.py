import torch


def apply_grad_mask(ctx, grads):
    if isinstance(grads, torch.Tensor):
        grads = (grads,)
    return tuple(g if nig else None for g, nig in zip(grads, ctx.needs_input_grad))


def broadcast_arguments(*args, to_contiguous: bool = True):
    last_dims = [t.shape[-1] for t in args]
    batch_dims = list(torch.broadcast_shapes(*[t.shape[:-1] for t in args]))

    result = [
        torch.broadcast_to(args[i], batch_dims + [last_dims[i]])
        for i in range(len(args))
    ]
    if to_contiguous:
        result = [x.contiguous() for x in result]

    return result


def last_dim_ctg(t: torch.Tensor):
    return (t if t.stride(-1) == 1 else t.contiguous()) if t is not None else None

def mtx_last_dim_ctg(t: torch.Tensor):
    return t if ((t.stride(-1) == 1) and (t.stride(-2) == t.size(-1))) else t.contiguous()
