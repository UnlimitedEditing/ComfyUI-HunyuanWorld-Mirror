"""
Camera-trajectory interpolation + real gsplat-rendered flythrough, ported from
Tencent's official HunyuanWorld-Mirror demo (src/utils/render_utils.py --
fetched and ported directly).

This is the actual quality gap between "export a raw Gaussian PLY and hope a
third-party WebGL viewer interprets it correctly" and what the official demo
does: it renders real video frames using the model's own gsplat rasterizer
(`WorldMirror.gs_renderer`, the same renderer instance HWMInference already
built), not a static point dump.

Simplifications made porting this into a single ComfyUI node call (each is a
deliberate, noted scope cut, not an oversight):
- The official `render_interpolated_video` writes an .mp4 directly via
  moviepy. This module instead returns RGB/depth-vis frames as tensors
  ([N, H, W, 3], float [0,1]) so a ComfyUI graph can feed them into
  VHS_VideoCombine (already a proven pattern elsewhere in this project) rather
  than adding moviepy as a new pip dependency for one function.
- The official function has a separate ~320-frame "intro hold" pre-roll where
  the camera stays fixed at the first pose while the Spread effect plays out
  once, before the camera starts moving along the real trajectory. This is
  dropped for simplicity; the effect (when enabled) instead plays continuously
  across the whole main trajectory (still a real animated flourish, just
  without the separate static-camera intro beat first).
- The official function's `runner`-cache fast path (an interactive-Gradio-only
  optimization for reusing a live rasterizer instance) is dropped entirely --
  always takes the plain `rasterize_batches(...)` call, which is what that
  fast path falls back to via its own `except` anyway.
- RGB2SH/SH2RGB round-tripping around the effects step is skipped: our
  workflow's Gaussians are exported/consumed as plain RGB (sh_degree=None,
  include_sh=False), and RGB2SH(x) -> SH2RGB(...) is the identity at SH
  degree 0, so it's mathematically a no-op here.
"""

from typing import Optional

import torch

from src.models.models.rasterization import GaussianSplatRenderer
from .gs_effects import GSEffects
from .color_map import apply_color_map_to_image


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3) -> quaternion (..., 4), wxyz (this module's own internal
    convention for trajectory interpolation only -- NOT the model's camera-pose
    decoding convention fixed elsewhere in this pack, which is scalar-last)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    mask2 = (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s

    return q


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def slerp_quaternions(q1: torch.Tensor, q2: torch.Tensor, t) -> torch.Tensor:
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    mask = dot < 0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.where(mask, -dot, dot)

    DOT_THRESHOLD = 0.9995
    mask_linear = dot > DOT_THRESHOLD
    result = torch.zeros_like(q1)

    if mask_linear.any():
        result_linear = q1 + t * (q2 - q1)
        norm = torch.norm(result_linear, dim=-1, keepdim=True)
        result_linear = result_linear / norm
        result = torch.where(mask_linear, result_linear, result)

    mask_slerp = ~mask_linear
    if mask_slerp.any():
        theta_0 = torch.acos(torch.abs(dot))
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        result_slerp = (s0 * q1) + (s1 * q2)
        result = torch.where(mask_slerp, result_slerp, result)

    return result


def _build_interpolated_traj(camtoworlds, intrinsics, index, nums):
    b = camtoworlds.shape[0]
    exts, ints = [], []
    tmp_camtoworlds = camtoworlds[:, index]
    tmp_intrinsics = intrinsics[:, index]
    for i in range(len(index) - 1):
        exts.append(tmp_camtoworlds[:, i:i + 1])
        ints.append(tmp_intrinsics[:, i:i + 1])
        R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
        R1, t1 = tmp_camtoworlds[:, i + 1, :3, :3], tmp_camtoworlds[:, i + 1, :3, 3]
        q0 = rotation_matrix_to_quaternion(R0)
        q1 = rotation_matrix_to_quaternion(R1)

        for j in range(1, nums + 1):
            alpha = j / (nums + 1)
            t_interp = (1 - alpha) * t0 + alpha * t1
            q_interp = slerp_quaternions(q0, q1, alpha)
            R_interp = quaternion_to_rotation_matrix(q_interp)

            ext = torch.eye(4, device=R_interp.device, dtype=R_interp.dtype)[None].repeat(b, 1, 1)
            ext[:, :3, :3] = R_interp
            ext[:, :3, 3] = t_interp

            K0 = tmp_intrinsics[:, i]
            K1 = tmp_intrinsics[:, i + 1]
            K = (1 - alpha) * K0 + alpha * K1

            exts.append(ext[:, None])
            ints.append(K[:, None])

    exts = torch.cat(exts, dim=1)[:1]
    ints = torch.cat(ints, dim=1)[:1]
    return exts, ints


def _build_wobble_traj(camtoworlds, intrinsics, nums, delta):
    t = torch.linspace(0, 1, nums, dtype=torch.float32, device=camtoworlds.device)
    t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
    tf = torch.eye(4, dtype=torch.float32, device=camtoworlds.device)
    radius = delta * 0.15
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * t) * radius
    exts = camtoworlds @ tf
    ints = intrinsics.repeat(1, exts.shape[1], 1, 1)
    return exts, ints


def _depth_vis(d: torch.Tensor) -> torch.Tensor:
    valid = d > 0
    if valid.any():
        near = d[valid].float().quantile(0.01).log()
    else:
        near = torch.tensor(0.0, device=d.device)
    far = d.flatten().float().quantile(0.99).log()
    x = d.float().clamp(min=1e-9).log()
    x = 1.0 - (x - near) / (far - near + 1e-9)
    return apply_color_map_to_image(x, "turbo")


def render_gaussian_flythrough(
    gs_renderer: GaussianSplatRenderer,
    splats: dict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple,
    interp_per_pair: int = 20,
    loop_reverse: bool = True,
    apply_spread_effect: bool = False,
    effect_speed: float = 0.04,
):
    """
    Args:
        gs_renderer: model.gs_renderer (WorldMirror's GaussianSplatRenderer instance)
        splats: gaussians dict (means/quats/scales/opacities/colors), batch-of-1
        camtoworlds: [1, S, 4, 4], intrinsics: [1, S, 3, 3] -- HWMInference's real output
        hw: (height, width) of the source images
        interp_per_pair: interpolated frames inserted between each pair of real camera poses
            (multi-view), or per-quarter-orbit frames for the single-image wobble path
        loop_reverse: append the reversed sequence so the video loops seamlessly
        apply_spread_effect: play the "Spread" reveal effect (gs_effects.py) continuously
            across the flythrough -- the one cosmetic flourish that's actually wired up
            in the official source (see gs_effects.py's module docstring)
        effect_speed: how fast the effect's internal clock advances per frame

    Returns:
        (rgb_frames, depth_frames): both [N, H, W, 3] float tensors in [0, 1]
    """
    b, s, _, _ = camtoworlds.shape
    h, w = hw

    if s > 1:
        all_ext, all_int = _build_interpolated_traj(camtoworlds, intrinsics, list(range(s)), interp_per_pair)
    else:
        radius_source = splats["means"][0].median(dim=0).values.norm(dim=-1)[None]
        all_ext, all_int = _build_wobble_traj(camtoworlds, intrinsics, interp_per_pair * 12, radius_source)

    effects = GSEffects(start_time=0.0, end_time=10.0) if apply_spread_effect else None

    if effects is not None:
        try:
            pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
        except Exception:
            pruned_splats = splats
        shift = pruned_splats["means"][0].median(dim=0).values
        scale_factor = (pruned_splats["means"][0] - shift).abs().quantile(0.95, dim=0).max()

    rendered_rgbs, rendered_depths = [], []
    t = 0.0
    t_st, t_ed, loop_dir, ignore_scale = 0.0, 0.0, 1, False
    chunk = 40 if effects is None else 1

    for st in range(0, all_ext.shape[1], chunk):
        ed = min(st + chunk, all_ext.shape[1])
        if effects is not None:
            sample_gsplat = {
                "means": (pruned_splats["means"][0] - shift) / scale_factor,
                "quats": pruned_splats["quats"][0],
                "scales": pruned_splats["scales"][0] / scale_factor,
                "opacities": pruned_splats["opacities"][0],
                "colors": pruned_splats["colors"][0],
            }
            effects_splats, flag = effects.apply_effect(sample_gsplat, t, effect_type=2, ignore_scale=ignore_scale)
            if loop_dir < 0:
                t -= effect_speed
            else:
                t += effect_speed
            if flag is not None and flag.mean() < 0.01 and t_ed == 0:
                t_ed = t

            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                effects_splats["means"][None], effects_splats["quats"][None], effects_splats["scales"][None],
                effects_splats["opacities"][None], effects_splats["colors"][None],
                all_ext[:, st:ed].to(torch.float32), all_int[:, st:ed].to(torch.float32),
                width=w, height=h,
            )

            if t > all_ext.shape[1] * effect_speed + t_st - (t_ed - t_st) * 2 - 15 * effect_speed or t < t_st:
                loop_dir *= -1
                t = t_ed if loop_dir == -1 else t
        else:
            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1], splats["quats"][:1], splats["scales"][:1], splats["opacities"][:1],
                splats["colors"][:1],
                all_ext[:, st:ed].to(torch.float32), all_int[:, st:ed].to(torch.float32),
                width=w, height=h,
            )

        rendered_rgbs.append(colors)
        rendered_depths.append(depths)

    rgbs = torch.cat(rendered_rgbs, dim=1)[0]           # [N, H, W, 3]
    depths = torch.cat(rendered_depths, dim=1)[0, ..., 0]  # [N, H, W]

    rgb_frames = rgbs.clamp(0, 1)
    depth_frames = torch.stack([_depth_vis(d) for d in depths]).movedim(-3, -1).clamp(0, 1)  # [N, H, W, 3]

    if loop_reverse and rgb_frames.shape[0] > 1:
        rgb_frames = torch.cat([rgb_frames, rgb_frames.flip(0)[1:-1]], dim=0)
        depth_frames = torch.cat([depth_frames, depth_frames.flip(0)[1:-1]], dim=0)

    return rgb_frames, depth_frames
