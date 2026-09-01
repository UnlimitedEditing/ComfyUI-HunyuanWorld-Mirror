# Rotation utilities for quaternions and rotation matrices
# References:
#   https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
#
# CONFIRMED LIVE BUG then fix (2026-09-02): quat_to_rotmat/rotmat_to_quat below were
# a from-scratch reimplementation using scalar-FIRST (w,x,y,z) component order. The
# actual tencent/HunyuanWorld-Mirror pretrained weights were trained against Tencent's
# own official demo code (huggingface.co/spaces/tencent/HunyuanWorld-Mirror,
# src/models/utils/rotation.py, fetched and diffed directly), whose quat_to_rotmat
# docstring explicitly states "Quaternion Order: XYZW or say ijkr, scalar-last".
# camera_utils.py (unchanged from official, confirmed byte-identical) feeds the
# cam_head's raw learned 9-dim output vector straight through rotmat_to_quat/
# quat_to_rotmat with NO reordering of its own -- so every element the pretrained
# camera-pose head learned to mean was being misread by a scalar-first formula here,
# producing a wrong rotation matrix for every predicted camera pose on every single
# inference run (not just multi-view: worldmirror.py calls this in the default,
# no-prior path via transform_camera_vector, forward() lines ~176-181). This is a
# strong root-cause candidate for the "wildly inaccurate, even on a single image"
# quality gap versus the official demo -- camera pose feeds directly into how every
# point/Gaussian gets placed in world space. Replaced below with Tencent's official
# implementation verbatim (same function names/signatures, so camera_utils.py and
# every other call site needed zero changes).

import torch
import torch.nn.functional as F


def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part (last, scalar-last convention) is non negative.
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        rotmat: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if rotmat.size(-1) != 3 or rotmat.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rotmat.shape}.")

    batch_dim = rotmat.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(rotmat.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Args:
        q1, q2: Quaternions of shape (..., 4) in wxyz format

    Returns:
        Product quaternion of shape (..., 4)
    """
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate.

    Args:
        q: Quaternion of shape (..., 4) in wxyz format

    Returns:
        Conjugate quaternion of shape (..., 4)
    """
    w, x, y, z = torch.unbind(q, -1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion inverse.

    Args:
        q: Quaternion of shape (..., 4) in wxyz format

    Returns:
        Inverse quaternion of shape (..., 4)
    """
    # For unit quaternions, inverse = conjugate
    q_conj = quat_conjugate(q)
    q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    return q_conj / q_norm_sq
