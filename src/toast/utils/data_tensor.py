from typing import Iterable
from types import EllipsisType

import torch


#: Inject ``__slots__`` into every subclass.  Benchmarking switch; default on.
ENABLE_SLOTS = True


# Module-level aliases: these are looked up once per call instead of walking
# the builtins/globals chain for ``torch.Tensor``.
_T = torch.Tensor
_new = object.__new__


def apply_method(fn_name):
    """Build a container method that forwards ``fn_name`` to every buffer.

    The returned function walks the subclass's precomputed buffer tuple and
    writes the results onto a freshly allocated container, i.e. it does the
    same work as ``self.apply(lambda x: getattr(x, fn_name)(*args, **kwargs))``
    without allocating the lambda or paying for the ``apply``/``named_apply``
    call frames.
    """
    def method(self, *args, **kwargs):
        cls = type(self)
        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(self, name, None)
            setattr(out, name, None if value is None
                    else getattr(value, fn_name)(*args, **kwargs))
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    method.__name__ = fn_name
    method.__qualname__ = "DataTensor." + fn_name
    if hasattr(torch.Tensor, fn_name):
        method.__doc__ = getattr(torch.Tensor, fn_name).__doc__
    return method


def apply_self_method(fn):
    """Build an in-place container method from a tensor function."""
    def method(self, *args, **kwargs):
        for name in type(self)._allbuf:
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, fn(value, *args, **kwargs))
        return self
    method.__doc__ = fn.__doc__
    return method


class DataTensorMeta(type):
    """Freezes each subclass's buffer layout and gives it ``__slots__``.

    Runs once per subclass, at class-creation time.  Nothing here happens at
    call time.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        def resolve(key):
            """Read *key* from the class body, else inherit it from a base."""
            if key in namespace:
                return list(namespace[key])
            for base in bases:
                inherited = getattr(base, key, None)
                if inherited:
                    return list(inherited)
            return []

        buffers = resolve("_buffers")
        optional = resolve("_optional_buffers")
        non_tensor = resolve("_non_tensor_data")

        if ENABLE_SLOTS and "__slots__" not in namespace:
            already_slotted = set()
            for base in bases:
                for klass in getattr(base, "__mro__", ()):
                    already_slotted.update(getattr(klass, "__slots__", ()) or ())
            slots = [
                n for n in buffers + optional + non_tensor
                if n not in already_slotted and n not in namespace
            ]
            # Slot the declared buffers -- that is what makes attribute access
            # cheap -- but keep a __dict__ as well, so a container still accepts
            # arbitrary attributes like any other Python object.  The dict is
            # created lazily, so instances that never take one stay small.
            # It may appear only once in a hierarchy ("__dict__ slot disallowed:
            # we already got one"), hence the check against the bases.
            if not any(
                "__dict__" in (getattr(klass, "__slots__", ()) or ())
                or "__dict__" in getattr(klass, "__dict__", {})
                for base in bases for klass in getattr(base, "__mro__", ())
                if klass is not object
            ):
                slots.append("__dict__")
            namespace["__slots__"] = tuple(slots)

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Buffer name tuples, resolved through the MRO exactly once.  Every
        # method below iterates these instead of rebuilding lists per call.
        cls._bufnames = tuple(buffers)                       # required buffers
        cls._nonnames = tuple(non_tensor)                    # non-tensor fields
        cls._allbuf = tuple(buffers + optional)              # every buffer
        cls._allnames = tuple(buffers + optional + non_tensor)
        cls._tailnames = tuple(optional + non_tensor)        # keyword-only tail
        cls._validnames = frozenset(cls._allnames)
        cls._buf0name = buffers[0] if buffers else None      # primary buffer
        return cls


class DataTensor(metaclass=DataTensorMeta):
    """Base class for batched tensor containers with a fixed last dimension.

    Drop-in replacement for :class:`toast.utils.data_tensor.DataTensor`; see
    that class for the full narrative documentation.  The public API, the
    semantics of every method, and the shape convention (:attr:`shape` is the
    *batch* shape — the primary buffer's shape without its last dimension) are
    unchanged.
    """

    __slots__ = ("__weakref__",)

    #: Move all buffers to CPU. Mirrors :meth:`torch.Tensor.cpu`.
    cpu = apply_method("cpu")
    #: Move all buffers to the default CUDA device. Mirrors :meth:`torch.Tensor.cuda`.
    cuda = apply_method("cuda")

    #: Cast all buffers to ``torch.uint8``.
    byte = apply_method("byte")
    #: Cast all buffers to ``torch.int16``.
    short = apply_method("short")
    #: Cast all buffers to ``torch.int64``.
    long = apply_method("long")
    #: Cast all buffers to ``torch.float16``.
    half = apply_method("half")
    #: Cast all buffers to ``torch.float32``.
    float = apply_method("float")
    #: Cast all buffers to ``torch.float64``.
    double = apply_method("double")

    #: Transfer all buffers to a device/dtype. Mirrors :meth:`torch.Tensor.to`.
    to = apply_method("to")
    #: Return a deep copy of all buffers. Mirrors :meth:`torch.Tensor.clone`.
    clone = apply_method("clone")
    #: Return a contiguous copy of all buffers.
    contiguous = apply_method("contiguous")
    #: Detach all buffers from the autograd graph.
    detach = apply_method("detach")

    _buffers: list[str] = []
    _optional_buffers: list[str] = []
    _non_tensor_data: list[str] = []

    # ------------------------------------------------------------------
    # Tensor metadata
    #
    # These read the primary buffer directly.  The original routed every one
    # of them through ``self._buf0()``, adding a call frame and a string
    # ``getattr`` on top of the property frame.
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device of the primary buffer."""
        return getattr(self, self._buf0name).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the primary buffer."""
        return getattr(self, self._buf0name).dtype

    @property
    def shape(self) -> torch.Size:
        """Batch shape — the shape of the primary buffer *excluding* its last dimension."""
        b = getattr(self, self._buf0name)
        return b.shape[:-1] if isinstance(b, _T) else b.shape

    @property
    def dim(self) -> int:
        """Number of batch dimensions (``len(self.shape)``)."""
        b = getattr(self, self._buf0name)
        return (b.ndim - 1) if isinstance(b, _T) else b.dim

    @property
    def numel(self) -> int:
        """Number of batch elements (product of :attr:`shape`)."""
        b = getattr(self, self._buf0name)
        return (b.numel() // b.shape[-1]) if isinstance(b, _T) else b.numel

    @property
    def requires_grad(self) -> bool:
        """``True`` if the primary buffer requires gradients."""
        return getattr(self, self._buf0name).requires_grad

    def size(self, dim: int = None) -> "torch.Size | int":
        """Return the batch shape, or a single batch dimension size.

        Args:
            dim: If given, return the size of that batch dimension.

        Returns:
            torch.Size | int: Full shape or a single dimension size.
        """
        shape = self.shape
        return shape if dim is None else shape[dim]

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------

    def requires_grad_(self, mode: bool) -> "DataTensor":
        """Set ``requires_grad`` on all buffers in-place.

        Args:
            mode: Desired gradient-tracking state.

        Returns:
            DataTensor: ``self``, for chaining.
        """
        for name in type(self)._allbuf:
            value = getattr(self, name, None)
            if value is not None:
                value.requires_grad_(mode)
        return self

    def buffers(self, requires_grad_only: bool = False) -> list[torch.Tensor]:
        """Collect all underlying tensors.

        Args:
            requires_grad_only: If ``True``, return only tensors that
                require gradients.

        Returns:
            list[torch.Tensor]: Flat list of all buffer tensors.
        """
        out = []
        for name in type(self)._allbuf:
            value = getattr(self, name, None)
            if value is None:
                continue
            if isinstance(value, _T):
                out.append(value)
            else:
                out += value.buffers()
        return [x for x in out if x.requires_grad] if requires_grad_only else out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def unsqueeze(self, dim: int) -> "DataTensor":
        """Insert a size-1 batch dimension at position *dim*.

        Args:
            dim: Batch dimension index (negative indexing supported).

        Returns:
            DataTensor: View with an extra dimension.
        """
        return self.apply(
            lambda x: x.unsqueeze(dim - 1 if dim < 0 else dim),
            lambda x: x.unsqueeze(dim)
        )

    def squeeze(self, dim: int) -> "DataTensor":
        """Remove a size-1 batch dimension at position *dim*.

        Args:
            dim: Batch dimension index to squeeze.

        Returns:
            DataTensor: View with the dimension removed (if it was size 1).
        """
        return self.apply(
            lambda x: x.squeeze(dim - 1 if dim < 0 else dim),
            lambda x: x.squeeze(dim)
        )

    def view(self, *args) -> "DataTensor":
        """Reshape the batch dimensions.

        Args:
            *args: New batch shape (integers or a single sequence).

        Returns:
            DataTensor: View with the new batch shape.
        """
        dim_lst = args[0] if isinstance(args[0], Iterable) else [*args]
        return self.apply(
            lambda x: x.view(*dim_lst, x.shape[-1]),
            lambda x: x.view(*dim_lst)
        )

    def reshape(self, *args) -> "DataTensor":
        """Reshape the batch dimensions (copies if necessary).

        Args:
            *args: New batch shape (integers or a single sequence).

        Returns:
            DataTensor: Tensor with the new batch shape.
        """
        dim_lst = args[0] if isinstance(args[0], Iterable) else [*args]
        return self.apply(
            lambda x: x.reshape(*dim_lst, x.shape[-1]),
            lambda x: x.reshape(*args)
        )

    def flatten(self) -> "DataTensor":
        """Flatten all batch dimensions into a single dimension.

        Returns:
            DataTensor: 1-D batch of elements.
        """
        return self.apply(lambda x: x.reshape(-1, x.shape[-1]), lambda x: x.flatten())

    def permute(self, *dims) -> "DataTensor":
        """Permute the batch dimensions.

        Args:
            *dims: Desired ordering of batch dimensions.

        Returns:
            DataTensor: View with permuted batch dimensions.
        """
        self._dim_check(dims)
        return self.apply(
            lambda x: x.permute(*dims, x.dim() - 1),
            lambda x: x.permute(*dims)
        )

    def flip(self, dims: list[int]) -> "DataTensor":
        """Flip the container along specified batch dimensions.

        Args:
            dims: Batch dimension indices to flip.

        Returns:
            DataTensor: Flipped view.
        """
        self._dim_check(dims)
        return self.apply(lambda x: x.flip(dims))

    def expand(self, *args) -> "DataTensor":
        """Expand to a larger batch shape (broadcasting, no copy).

        Args:
            *args: Target batch shape.

        Returns:
            DataTensor: Expanded view.
        """
        return self.apply(lambda x: x.expand(*args, -1), lambda x: x.expand(*args))

    def expand_as(self, other: "DataTensor") -> "DataTensor":
        """Expand to match the batch shape of *other*.

        Args:
            other: Reference container.

        Returns:
            DataTensor: Expanded view.
        """
        return self.named_apply(lambda name, value: value.expand_as(getattr(other, name)))

    def broadcast_to(self, shape) -> "DataTensor":
        """Broadcast to a given batch shape.

        Args:
            shape: Target batch shape.

        Returns:
            DataTensor: Broadcast view.
        """
        s = list(shape)
        return self.apply(
            lambda x: x.broadcast_to(s + [x.shape[-1]]),
            lambda x: x.broadcast_to(shape)
        )

    def repeat(self, *repeats) -> "DataTensor":
        """Repeat the container along batch dimensions.

        Args:
            *repeats: Number of repetitions per batch dimension.

        Returns:
            DataTensor: Repeated tensor.
        """
        return self.apply(
            lambda x: x.repeat(*repeats, 1),
            lambda x: x.repeat(*repeats)
        )

    def repeat_interleave(self, repeats, dim: int = None) -> "DataTensor":
        """Repeat elements along a batch dimension.

        Args:
            repeats: Number of repetitions per element (int or 1-D tensor).
            dim: Batch dimension to repeat along.

        Returns:
            DataTensor: Tensor with repeated elements.
        """
        assert dim < self.dim
        return self.apply(lambda x: x.repeat_interleave(repeats, dim))

    def take_along_dim(self, dim: int, indices: torch.Tensor) -> "DataTensor":
        """Gather elements along a batch dimension using an index tensor.

        Args:
            dim: Batch dimension to gather along.
            indices: Long tensor of indices.

        Returns:
            DataTensor: Gathered elements.
        """
        return self.apply(
            lambda x: x.take_along_dim(dim=dim - 1 if dim < 0 else dim, indices=indices.unsqueeze(-1)),
            lambda x: x.take_along_dim(dim=dim, indices=indices)
        )

    def view_as(self, other: "DataTensor") -> "DataTensor":
        """View with the batch shape of *other*.

        Args:
            other: Reference container.

        Returns:
            DataTensor: View matching *other*'s batch shape.
        """
        return self.named_apply(lambda name, value: value.view_as(getattr(other, name)))

    def reshape_as(self, other: "DataTensor") -> "DataTensor":
        """Reshape to match the batch shape of *other*.

        Args:
            other: Reference container.

        Returns:
            DataTensor: Reshaped copy matching *other*'s batch shape.
        """
        return self.named_apply(lambda name, value: value.reshape_as(getattr(other, name)))

    # ------------------------------------------------------------------
    # Combining
    # ------------------------------------------------------------------

    @staticmethod
    def cat(lst: "list[DataTensor]", dim: int = 0) -> "DataTensor":
        """Concatenate a list of containers along a batch dimension.

        Mirrors :func:`torch.cat`.  All elements must be the same subclass
        with the same feature size.

        Args:
            lst: Containers to concatenate.
            dim: Batch dimension to concatenate along.

        Returns:
            DataTensor: Concatenated container of the same subclass.

        Example::

            q = Quat.cat([q1, q2, q3], dim=0)
        """
        first = lst[0]
        cls = type(first)
        ndim = first.dim
        if dim >= 0 and dim > ndim:
            raise IndexError("Dimension out of range. Valid dimensions: [{}; {}]".format(
                -ndim, ndim - 1
            ))
        tensor_dim = dim - 1 if dim < 0 else dim

        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(first, name, None)
            if value is None:
                setattr(out, name, None)
            elif isinstance(value, _T):
                setattr(out, name, torch.cat([getattr(x, name) for x in lst], dim=tensor_dim))
            else:
                setattr(out, name, type(value).cat([getattr(x, name) for x in lst], dim=dim))
        for name in cls._nonnames:
            setattr(out, name, getattr(first, name, None))
        return out

    @staticmethod
    def stack(lst: "list[DataTensor]", dim: int = 0) -> "DataTensor":
        """Stack a list of containers along a *new* batch dimension.

        Mirrors :func:`torch.stack`.  All elements must be the same subclass
        with identical batch shape and feature size.

        Args:
            lst: Containers to stack.
            dim: Position of the new batch dimension.

        Returns:
            DataTensor: Stacked container with one extra batch dimension.

        Example::

            q = Quat.stack([q1, q2], dim=0)  # shape (2, *q1.shape)
        """
        first = lst[0]
        cls = type(first)
        ndim = first.dim
        if dim >= 0 and dim > ndim:
            raise IndexError("Dimension out of range. Valid dimensions: [{}; {}]".format(
                -ndim, ndim
            ))
        tensor_dim = dim - 1 if dim < 0 else dim

        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(first, name, None)
            if value is None:
                setattr(out, name, None)
            elif isinstance(value, _T):
                setattr(out, name, torch.stack([getattr(x, name) for x in lst], dim=tensor_dim))
            else:
                setattr(out, name, type(value).stack([getattr(x, name) for x in lst], dim=dim))
        for name in cls._nonnames:
            setattr(out, name, getattr(first, name, None))
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a dictionary of detached buffer copies, suitable for saving.

        Returns:
            dict: Mapping from buffer name to cloned tensor (or nested dict
            for nested :class:`DataTensor` buffers).
        """
        cls = type(self)
        d = {}
        for name in cls._allbuf:
            value = getattr(self, name, None)
            if value is None:
                # an optional buffer the instance does not carry -- skip it,
                # exactly as to_dict() does
                continue
            if isinstance(value, _T):
                d[name] = value.detach().clone()
            else:
                d[name] = value.state_dict()
        for name in cls._nonnames:
            if hasattr(self, name):
                d[name] = getattr(self, name)
        return d

    def to_dict(self) -> dict:
        """Serialize all buffers to Python lists (JSON-compatible).

        Returns:
            dict: Nested dict with all tensor data converted via ``.tolist()``.
        """
        cls = type(self)
        result = {}
        for name in cls._allbuf:
            value = getattr(self, name, None)
            if value is None:
                continue
            result[name] = value.tolist() if isinstance(value, _T) else value.to_dict()
        for name in cls._nonnames:
            value = getattr(self, name, None)
            if value is not None:
                result[name] = value
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buf0(self):
        return getattr(self, self._buf0name)

    def _dim_check(self, dims):
        ndim = self.dim
        for dim in dims:
            if dim >= ndim:
                raise IndexError(
                    "Invalid dimension given: {} (ndims: {})".format(dim, ndim)
                )

    def apply_(self, fn):
        """Apply *fn* to every buffer in-place and return ``self``."""
        for name in type(self)._allbuf:
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, fn(value))
        return self

    def apply(self, fn, dt_fn=None):
        """Map *fn* over every buffer and return a new container.

        Args:
            fn: Applied to :class:`torch.Tensor` buffers — and, when *dt_fn*
                is ``None``, to nested :class:`DataTensor` buffers too.
            dt_fn: If given, used for nested :class:`DataTensor` buffers.
        """
        cls = type(self)
        out = _new(cls)
        if dt_fn is None:
            for name in cls._allbuf:
                value = getattr(self, name, None)
                setattr(out, name, None if value is None else fn(value))
        else:
            for name in cls._allbuf:
                value = getattr(self, name, None)
                if value is None:
                    setattr(out, name, None)
                else:
                    setattr(out, name, fn(value) if isinstance(value, _T) else dt_fn(value))
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    def named_apply(self, fn, dt_fn=None):
        """Like :meth:`apply`, but *fn* / *dt_fn* also receive the buffer name."""
        cls = type(self)
        out = _new(cls)
        if dt_fn is None:
            for name in cls._allbuf:
                value = getattr(self, name, None)
                setattr(out, name, None if value is None else fn(name, value))
        else:
            for name in cls._allbuf:
                value = getattr(self, name, None)
                if value is None:
                    setattr(out, name, None)
                else:
                    setattr(out, name,
                            fn(name, value) if isinstance(value, _T) else dt_fn(name, value))
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    def __init__(self, *args, **kwargs):
        cls = type(self)
        allnames = cls._allnames
        if len(args) + len(kwargs) > len(allnames):
            raise ValueError("Too many arguments")
        unknown = set(kwargs.keys()).difference(cls._validnames)
        if len(unknown) > 0:
            raise ValueError("Unexpected arguments: {}".format(', '.join(unknown)))
        bufnames = cls._bufnames
        missing = set(bufnames[len(args):]).difference(set(kwargs))
        if len(missing) > 0:
            raise ValueError("Missing arguments: {}".format(', '.join(missing)))
        for name, value in zip(allnames, args):
            setattr(self, name, value)
        for name in bufnames:
            if getattr(self, name, None) is None:
                setattr(self, name, kwargs[name])
        for name in cls._tailnames:
            if not hasattr(self, name):
                setattr(self, name, kwargs.get(name, None))

    def __len__(self):
        shape = self.shape
        return shape[0] if len(shape) > 0 else 1

    def __getitem__(self, item):
        # A trailing ``slice(None)`` keeps the feature dimension intact when
        # the index contains an ellipsis.
        item_t = item
        if isinstance(item, tuple):
            for element in item:
                if isinstance(element, EllipsisType):
                    item_t = tuple(list(item) + [slice(None, None, None)])
                    break

        cls = type(self)
        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(self, name, None)
            if value is None:
                setattr(out, name, None)
            else:
                setattr(out, name, value[item_t] if isinstance(value, _T) else value[item])
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    def __setitem__(self, key, value):
        for name in type(self)._allbuf:
            target = getattr(self, name, None)
            if target is not None:
                target[key] = getattr(value, name)

    def __add__(self, other: "DataTensor"):
        cls = type(self)
        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(self, name, None)
            setattr(out, name, None if value is None else value + getattr(other, name))
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    def __sub__(self, other: "DataTensor"):
        cls = type(self)
        out = _new(cls)
        for name in cls._allbuf:
            value = getattr(self, name, None)
            setattr(out, name, None if value is None else value - getattr(other, name))
        for name in cls._nonnames:
            setattr(out, name, getattr(self, name, None))
        return out

    def __repr__(self):
        r = type(self).__name__ + '(\n'
        for name in type(self)._allnames:
            r += '    {}={},\n'.format(name, repr(getattr(self, name)).replace('\n', '\n    '))
        r += ')'
        return r

    # ------------------------------------------------------------------
    # Pickling — needed because ``__slots__`` removes ``__dict__``.
    # ------------------------------------------------------------------

    def __getstate__(self):
        return {name: getattr(self, name, None) for name in type(self)._allnames}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
