"""
Depth/normal discontinuity ("flying pixel") edge masking, ported verbatim from
Tencent's own official HunyuanWorld-Mirror Gradio demo
(huggingface.co/spaces/tencent/HunyuanWorld-Mirror, src/utils/geometry.py) --
this ComfyUI node pack (cedarconnor/ComfyUI-HunyuanWorld-Mirror) never had it.

The official demo applies this to points3d before export; it is not applied to
Gaussian splats there either (the model's own internal prune_gs voxel-merge,
src/models/models/rasterization.py, already dedupes/prunes splats and reorders
them away from a per-pixel grid, so this per-pixel mask can't be aligned back
onto exported splats -- only onto points3d/depth/normals, which still carry a
[frame, H, W] grid at export time).
"""

import warnings
from contextlib import contextmanager
from numbers import Number
from typing import Tuple, Union

import numpy as np


@contextmanager
def _no_warnings(category=RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1:],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1:],
        x.strides[axis],
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int, ...], stride: Tuple[int, ...], axis: Tuple[int, ...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    return np.nanmax(a_sliding, axis=-1)


def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], axis: Tuple[int, ...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    return max_pool_nd(x, kernel_size, stride, padding, tuple(axis))


def depth_edge(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """Pixels whose neighbors have a large difference in depth (silhouette/discontinuity edges)."""
    with _no_warnings():
        if mask is None:
            diff = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + \
                   max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
        else:
            diff = max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + \
                   max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2)

        edge = np.zeros_like(depth, dtype=bool)
        if atol is not None:
            edge |= diff > atol
        if rtol is not None:
            edge |= diff / depth > rtol
        return edge


def normals_edge(normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """Pixels whose neighbors have a large difference in surface normal direction."""
    assert normals.ndim >= 3 and normals.shape[-1] == 3, "normal should be of shape (..., height, width, 3)"
    with _no_warnings():
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

        padding = kernel_size // 2
        normals_window = sliding_window_2d(
            np.pad(normals, (*([(0, 0)] * (normals.ndim - 3)), (padding, padding), (padding, padding), (0, 0)), mode="edge"),
            window_size=kernel_size, stride=1, axis=(-3, -2),
        )
        if mask is None:
            angle_diff = np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)).max(axis=(-2, -1))
        else:
            mask_window = sliding_window_2d(
                np.pad(mask, (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)), mode="edge"),
                window_size=kernel_size, stride=1, axis=(-3, -2),
            )
            angle_diff = np.where(mask_window, np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)), 0).max(axis=(-2, -1))

        angle_diff = max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
        return angle_diff > np.deg2rad(tol)
