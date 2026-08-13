r"""Unified trajectory-interpolation operations dispatched to the active backend.

A *trajectory* is a discrete sequence of timestamped SE(3) keyframes.
These operations sample the trajectory at arbitrary query times, interpolating
between keyframes (SLERP for rotation, LERP for translation) and optionally
extrapolating beyond the sequence boundaries.

"""

from typing import Literal, Optional

import torch

from .backend import get_backend


def all_same_device(tensor_lst: list[torch.Tensor]):
    tensor_lst = filter(lambda x: x is not None, tensor_lst)
    device_lst = set([x.device for x in tensor_lst])
    assert len(device_lst) == 1, (
        "All tensors must be on the same device, but got: {}".format(device_lst))


def sample_trajectory(
    query_times: torch.Tensor,
    seq_times: torch.Tensor,
    seq_quats: Optional[torch.Tensor] = None,
    seq_t: Optional[torch.Tensor] = None,
    seq_mask: Optional[torch.Tensor] = None,
    use_only_valid_keyframes: bool = True,
    extrapolate: bool = True,
    extrapolation_window: int = 1,
    mask_mode: Literal["and", "or"] = "and",
    return_velocities: bool = False,
    return_indices: bool = False,
    return_mask: bool = False,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    r"""Sample a trajectory at given query times using pose interpolation.

    For each query time :math:`t_q`, the two valid closest bracketing keyframes
    at times :math:`t_L \le t_q` and :math:`t_R \ge t_q` are found and the pose
    is interpolated:

    .. math::

        \alpha = \frac{t_q - t_L}{t_R - t_L}, \qquad
        q(t_q) = \operatorname{slerp}(q_L,\, q_R,\, \alpha), \qquad
        \mathbf{t}(t_q) = (1 - \alpha)\,\mathbf{t}_L + \alpha\,\mathbf{t}_R
    
    If for some reason :math:`t_L = t_R`, :math:`\alpha = 0.5` is used, and
    gradients w.r.t. :math:`t_q, t_L, t_R` are zero.
    
    **Masking.** Individual keyframes can be marked invalid via ``seq_mask``
    set to ``False`` for this particular keyframe. When 
    ``use_only_valid_keyframes=True`` only valid keyframes participate
    in bracketing and boundary selection. If no ``seq_mask`` provided, all
    keyframes are considered to be valid. If no valid keyframes exist for 
    some batch element, resulting quaterions are unit, translation and velocities
    are zero. 

    **Extrapolation.** When the query falls outside the sequence, the two
    nearest valid boundary keyframes (selected by ``extrapolation_window``)
    are used with the same formula, which becomes linear extrapolation. If
    enough valid keyframes exist, :math:`|R-L| = extrapolation\_window`. If 
    sequence has less valid keyframes than neccessary, two furthermost valid 
    keyframes are chosen.

    **Batch layout**:

    .. math::

        \begin{array}{ll}
        \texttt{seq\_times}  & (\ldots, K) \\
        \texttt{seq\_quats}  & (\ldots, K, 4) \\
        \texttt{seq\_t}      & (\ldots, K, 3) \\
        \texttt{seq\_mask}   & (\ldots, K) \\
        \texttt{query\_times}& (\ldots, Q) \\
        \text{outputs}       & (\ldots, Q, \cdot)
        \end{array}

    where :math:`\ldots` is any common batch prefix.
    
    .. attention::
    	``seq_times`` must be sorted in non-decreasing order along the last
    	dimension; behaviour is undefined otherwise, and this is not verified.
    	``query_times`` may be in any order -- each query is bracketed by an
    	independent binary search.
    
    .. caution::
    	Using ``seq_times`` with difference between neighbouring keyframes 
    	close to dtype epsilon will produce unstable velocity estimations 
    	and would result in unstable gradients w.r.t. velocities, ``seq_times`` 
    	and ``query_times`` at backward.

    Args:
        query_times: Query timestamps, shape :math:`(\ldots, Q)`. Any order. 
        seq_times: Keyframe timestamps (monotonically increasing along the
            last dimension), shape :math:`(\ldots, K)`.
        seq_quats: Keyframe quaternions (WXYZ), shape :math:`(\ldots, K, 4)`.
            Pass ``None`` to skip rotation interpolation.
        seq_t: Keyframe translations, shape :math:`(\ldots, K, 3)`.
            Pass ``None`` to skip translation interpolation.
        seq_mask: Boolean keyframe validity mask, shape :math:`(\ldots, K)`.
            ``True`` means valid.  If ``None``, all keyframes are valid.
        use_only_valid_keyframes: When ``True`` (default), invalid keyframes
            are excluded from bracketing and boundary search.
        extrapolate: When ``True`` (default), queries outside the sequence
            range are extrapolated from the nearest valid boundary pair.
            When ``False``, clamped to the boundary keyframe pose.
        extrapolation_window: Number of steps used to define the extrapolation
            direction — the :math:`k`-th valid frame from each end forms the
            second point of the extrapolation pair.  Default is ``1``.
        mask_mode: How to combine keyframe masks when computing the output
            validity mask (only relevant with ``return_mask=True``).

            - ``'and'``: output valid only if **both** bracketing keyframes
              are valid.
            - ``'or'``:  output valid if **at least one** bracketing keyframe
              is valid.
        return_velocities: If ``True``, also return the constant-segment
            linear velocity :math:`v = (t_R - t_L) / (t_R - t_L)` and
            angular velocity 
            :math:`\omega = \operatorname{axis\_angle}(q_R\,q_L^{-1}) / \Delta t`.
        return_indices: If ``True``, also return the left and right keyframe
            indices for each query as an integer tensor of shape
            :math:`(\ldots, Q, 2)`.
        return_mask: If ``True``, also return a boolean validity mask of
            shape :math:`(\ldots, Q)` indicating which query outputs are
            interpolated from valid keyframes.

    Returns:
        6-tuple ``(out_quats, out_t, out_mask, out_v, out_w, out_indices)``:

        - ``out_quats``: Interpolated quaternions :math:`(\ldots, Q, 4)`.
          ``None`` if ``seq_quats`` was ``None``.
        - ``out_t``: Interpolated translations :math:`(\ldots, Q, 3)`.
          ``None`` if ``seq_t`` was ``None``.
        - ``out_mask``: Boolean validity mask :math:`(\ldots, Q)`.
          ``None`` if ``return_mask=False``.
        - ``out_v``: Linear velocities :math:`(\ldots, Q, 3)`.
          ``None`` if ``return_velocities=False`` or ``seq_t=None``.
        - ``out_w``: Angular velocities (axis-angle) :math:`(\ldots, Q, 3)`.
          ``None`` if ``return_velocities=False`` or ``seq_quats=None``.
        - ``out_indices``: Keyframe indices :math:`(\ldots, Q, 2)`.
          ``None`` if ``return_indices=False``.

    Example::

        >>> import torch
        >>> from toast.ops.quat_ops import quat_unit
        >>> K = 3
        >>> seq_times = torch.tensor([0.0, 1.0, 2.0])
        >>> seq_quats = quat_unit(K)
        >>> seq_t = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        >>> out_q, out_t, *_ = sample_trajectory(
        ...     query_times=torch.tensor([0.5, 1.5]),
        ...     seq_times=seq_times,
        ...     seq_quats=seq_quats,
        ...     seq_t=seq_t,
        ... )
        >>> out_t  # tensor([[0.5, 0., 0.], [1.5, 0., 0.]])
    """
    all_same_device([query_times, seq_times, seq_quats, seq_t, seq_mask])

    return get_backend().sample_trajectory(
        query_times,
        seq_times,
        seq_quats,
        seq_t,
        seq_mask,
        use_only_valid_keyframes,
        extrapolate,
        extrapolation_window,
        mask_mode,
        return_velocities,
        return_indices,
        return_mask,
    )


