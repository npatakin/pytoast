Defining your own container
===========================

.. currentmodule:: toast.utils

A :class:`DataTensor` is a **collection of named tensors that share batch
dimensions**, where each tensor also carries a fixed-size **last dimension that
the container protects**. That trailing dimension is the payload of one element —
4 for a quaternion, 3 for a translation — and it takes no part in any operation
over dimensions: it is not counted in :attr:`~DataTensor.shape` or
:attr:`~DataTensor.dim`, and indexing, slicing, ``reshape``, ``permute``,
``cat``, ``stack`` and friends never see it.

Subclassing :class:`DataTensor` gives any such group of tensors that whole
tensor-like interface — indexing, ``cat``/``stack``, reshaping, device and dtype
moves — without writing any of it. Declare the buffer names and you are done:

.. code-block:: python

    import torch
    from toast.utils import DataTensor

    class Ray(DataTensor):
        _buffers = ['origin', 'direction']   # required, both (..., 3)
        _optional_buffers = ['color']        # may be absent on an instance
        _non_tensor_data = ['frame']         # carried along, never touched

        origin: torch.Tensor
        direction: torch.Tensor

Three class-level lists define the contract:

``_buffers``
    Required tensors, in order. The **first** is the primary buffer: it defines
    the batch shape and the device and dtype reported by the container.

``_optional_buffers``
    Tensors that may be ``None``. Every operation skips those that are unset.

``_non_tensor_data``
    Plain Python attributes — labels, frame names, metadata. They are copied to
    derived containers untouched, never moved between devices or indexed.

No ``__init__`` is needed. The inherited one accepts positional and keyword
arguments in ``_buffers + _optional_buffers + _non_tensor_data`` order, and
raises on missing, unknown or excess arguments:

.. code-block:: python

    r = Ray(torch.randn(4, 5, 3), torch.randn(4, 5, 3), frame='world')

    r.shape                  # (4, 5) -- batch only; the trailing 3 is the feature dim
    r.device, r.dtype        # taken from the primary buffer
    r[:, 2:4].shape          # (4, 2)
    Ray.stack([r, r]).shape  # (2, 4, 5)
    r.cuda().double()        # every buffer moves together
    r.frame                  # 'world' -- survives all of the above

Write your own ``__init__`` only when construction needs to normalise something.
:class:`~toast.transforms.SE3` does this to promote a raw tensor to a
:class:`~toast.transforms.Quat` and to put the translation on the rotation's
device and dtype.

.. important::

   Every buffer must carry a trailing fixed-size dimension, and all buffers must
   agree on the batch dimensions in front of it. A per-element **scalar** needs an
   explicit ``(..., 1)``, not ``(...)`` — this is why
   :attr:`~toast.transforms.LinearMotion.pose_time` is ``(..., 1)`` and
   interpolation weights are ``(..., 1)``.

A buffer may itself be a :class:`DataTensor` — that is how
:class:`~toast.transforms.SE3` holds a :class:`~toast.transforms.Quat`, and
:class:`~toast.transforms.LinearMotion` holds an :class:`~toast.transforms.SE3`.
Nesting is handled recursively.
