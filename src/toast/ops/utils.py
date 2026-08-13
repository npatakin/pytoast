import torch

from toast._C import broadcast_args as _broadcast_args


def broadcast_args(
        tensors: list[torch.Tensor],
        mtx_args: set[int] = set(),
        to_feature_contiguous: bool = True
) -> tuple[list[torch.Tensor], bool]:
    try:
        return _broadcast_args(
            tensors,
            mtx_args=mtx_args,
            to_feature_contiguous=to_feature_contiguous
        )
    except Exception as e:
        shapes = [tuple(t.shape) for t in tensors]
        raise ValueError(
            f"Cannot broadcast tensors together. Shapes: {shapes}"
        ) from e


def broadcast_args_torch(
        tensors: list[torch.Tensor], **kwargs
) -> list[torch.Tensor]:
    kwargs.setdefault('to_feature_contiguous', False)
    tensors, _ = broadcast_args(tensors, **kwargs)
    return tensors
