
Quat
----

:class:`Quat` represents a batch of 3-D rotations stored as **unit
quaternions in WXYZ order** (real part first).  It overloads ``@`` for
both rotation of points and composition of rotations.

.. currentmodule:: toast.transforms

.. autoclass:: Quat
   :members:
   :special-members: __matmul__, __neg__
   :show-inheritance:
   :member-order: bysource
