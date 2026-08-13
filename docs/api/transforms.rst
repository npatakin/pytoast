Object-Oriented Wrappers
========================

The :mod:`toast.transforms` module provides high-level, object-oriented
wrappers around the low-level functional operations.  Both classes inherit
from :class:`~toast.utils.DataTensor`, so all shape / device / dtype
utilities (``clone``, ``to``, ``cuda``, ``unsqueeze``, …) are available
out of the box.

.. currentmodule:: toast.transforms

.. toctree::
    :caption: Transforms
    :hidden:

    quat
    se3
    linear_motion
