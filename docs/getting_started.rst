Getting Started
================

Quick Start
-----------

Quaternions
~~~~~~~~~~~

:class:`~toast.transforms.Quat` represents a 3-D rotation.  Quaternions
are stored in **wxyz** order — real part first — in a ``torch.Tensor`` of
shape ``(..., 4)``.

.. code-block:: python

    import torch
    from toast.transforms import Quat

    # Sample a batch of random rotations
    q = Quat.random(1000, device="cuda", dtype=torch.float32)

    # Rotate a batch of 3-D points
    p = torch.randn(1000, 3, device="cuda")
    p_rotated = q @ p           # shape (1000, 3)

    # Compose two rotations
    q1 = Quat.random(1000, device="cuda")
    q2 = Quat.random(1000, device="cuda")
    q12 = q1 @ q2               # Quat, shape (1000,)

    # Invert and verify round-trip
    p_back = q.inv() @ p_rotated
    assert torch.allclose(p, p_back, atol=1e-5)

    # Convert representations
    axis_angle = q.std().to_axis_angle()   # (1000, 3)
    mtx        = q.to_matrix()            # (1000, 3, 3)
    q_back     = Quat.from_matrix(mtx)

    # Spherical interpolation
    t = torch.rand(1000, 1, device="cuda")
    q_mid = q1.lerp(q2, t)     # halfway between q1 and q2

SE(3) Transforms
~~~~~~~~~~~~~~~~

:class:`~toast.transforms.SE3` pairs a :class:`~toast.transforms.Quat`
with a translation vector to represent a rigid body transform.  Applying
``T`` to a point ``p`` computes ``q.rotate(p) + t``.

.. code-block:: python

    from toast.transforms import SE3, Quat

    q = Quat.random(4, device="cuda")
    t = torch.randn(4, 3, device="cuda")
    T = SE3(q, t)

    p = torch.randn(4, 3, device="cuda")
    p_out  = T @ p          # transform
    p_back = T.inv() @ p_out
    assert torch.allclose(p, p_back, atol=1e-5)

    # Compose two transforms
    T1 = SE3(Quat.random(4, device="cuda"), torch.randn(4, 3, device="cuda"))
    T2 = SE3(Quat.random(4, device="cuda"), torch.randn(4, 3, device="cuda"))
    T12 = T1 @ T2

    # Convert to / from homogeneous matrices
    M4x4 = T.to_matrix()          # (4, 4, 4)
    M3x4 = T.to_matrix(to_3x4=True)  # (4, 3, 4)
    T2   = SE3.from_matrix(M4x4)

    # Interpolate
    w = torch.rand(4, 1, device="cuda")
    T_mid = T1.lerp(T2, w)

Backend System
--------------

All transform operations are dispatched through a backend registry.  The
default backend is ``"cpp"``, which runs the fused kernels on whichever device
the tensors are on.  You can switch to the pure-PyTorch path to compare against
it or to get a readable stack trace:

.. code-block:: python

    from toast import use_backend

    a, b = Quat.random(1000), Quat.random(1000)
    weight = torch.rand(1000, 1)

    q_mid = a.lerp(b, weight)          # 'cpp' -- the fused kernels

    with use_backend("torch"):
        q_mid = a.lerp(b, weight)      # the reference implementation

Available backends:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Description
   * - ``"cpp"``
     - Fused CUDA/C++ kernels (default).  Supports non-contiguous tensors
       and both ``float32`` and ``float64``.
   * - ``"torch"``
     - Pure PyTorch reference implementation.  Useful for testing and
       CPU-only deployment.

Autograd Support
----------------

Both backends are fully differentiable. Gradients flow through all
operations, unless stated otherwise. 

.. code-block:: python

    q = Quat.random(10, device="cuda").data.requires_grad_(True)
    p = torch.randn(10, 3, device="cuda")

    loss = (Quat(q) @ p).sum()
    loss.backward()
    print(q.grad)   # gradient w.r.t. quaternion parameters
