import os

from .ops import *
from .transforms import *
from .utils import *


def get_include_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'include')


def has_cuda_support() -> bool:
    """Whether this build of the extension contains CUDA kernels.
    ``False`` for a CPU-only build (one made without the CUDA toolkit present,
    or forced with ``TOAST_CPU_ONLY=1``).
    """
    from . import _C
    return bool(getattr(_C, 'built_with_cuda', False))
