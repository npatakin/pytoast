
DataTensor
==========

:class:`~toast.utils.DataTensor` is the base class shared by
:class:`~toast.transforms.Quat` and :class:`~toast.transforms.SE3`.
It provides a tensor-like interface over batches of named buffers so that
all standard PyTorch device / dtype / shape operations work uniformly on
any subclass.

See :doc:`/custom_containers` for how to define one of your own.

.. currentmodule:: toast.utils

.. autoclass:: DataTensor
   :members:
   :show-inheritance:
   :member-order: bysource
