"""
Gaussian-splat render effects, ported from Tencent's official HunyuanWorld-Mirror
demo (huggingface.co/spaces/tencent/HunyuanWorld-Mirror, src/utils/gs_effects.py --
fetched and ported directly).

NOTE on scope, confirmed by reading the official source directly: `apply_effect()`
only actually implements `effect_type=2` ("Spread" -- a radial reveal where the
scene materializes outward from center with a color/opacity trail, plus random
per-point masking during the transition). `twister()` and `rain()` are defined
below (ported verbatim for completeness/future use) but are NEVER called by
`apply_effect()` in the official source -- there is no effect_type value that
routes to them. They are vestigial in the original codebase, not a hidden
feature we're missing. Ported here unchanged rather than silently dropped, but
flagged clearly so nobody assumes they're reachable.
"""

from math import floor

import numpy as np
import torch


def fract(x):
    """Get fractional part of a number"""
    if isinstance(x, torch.Tensor):
        return x - torch.floor(x)
    return x - floor(x)


class GSEffects:
    """Convert GLSL GS render effects to PyTorch - vectorized for batch processing"""

    def __init__(self, start_time=0.0, end_time=10.0):
        self.start_time = start_time
        self.end_time = end_time

    @staticmethod
    def smoothstep(edge0, edge1, x):
        if isinstance(x, torch.Tensor):
            result = torch.zeros_like(x, dtype=x.dtype)
            mask_low = x < edge0
            mask_high = x > edge1
            mask_mid = ~(mask_low | mask_high)

            t = (x[mask_mid] - edge0) / (edge1 - edge0)
            result[mask_mid] = t * t * (3.0 - 2.0 * t)
            result[mask_low] = 0.0
            result[mask_high] = 1.0
            return result
        else:
            if x < edge0:
                return 0.0
            if x > edge1:
                return 1.0
            t = (x - edge0) / (edge1 - edge0)
            return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def step(edge, x):
        if isinstance(x, torch.Tensor):
            return (x >= edge).to(x.dtype)
        if isinstance(edge, torch.Tensor):
            return (x >= edge).to(edge.dtype)
        return 1.0 if x >= edge else 0.0

    @staticmethod
    def mix(x, y, a):
        return x * (1.0 - a) + y * a

    @staticmethod
    def clamp(x, min_val, max_val):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        return max(min_val, min(max_val, x))

    @staticmethod
    def length_xz(pos):
        if pos.dim() == 1:
            return torch.sqrt(pos[0] ** 2 + pos[2] ** 2)
        return torch.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    @staticmethod
    def length_vec(v):
        if v.dim() == 1:
            return torch.sqrt(torch.sum(v ** 2))
        return torch.sqrt(torch.sum(v ** 2, dim=1))

    @staticmethod
    def hash(p):
        p = fract(p * 0.3183099 + 0.1)
        p = p * 17.0
        return torch.stack([
            fract(p[:, 0] * p[:, 1] * p[:, 2]),
            fract(p[:, 0] + p[:, 1] * p[:, 2]),
            fract(p[:, 0] * p[:, 1] + p[:, 2])
        ], dim=1)

    @staticmethod
    def noise(p):
        i = torch.floor(p).to(torch.long)
        f = fract(p)
        f = f * f * (3.0 - 2.0 * f)

        def get_hash_offset(offset):
            return GSEffects.hash(i.to(p.dtype) + offset)

        n000 = get_hash_offset(torch.tensor([0, 0, 0], dtype=p.dtype, device=p.device))
        n100 = get_hash_offset(torch.tensor([1, 0, 0], dtype=p.dtype, device=p.device))
        n010 = get_hash_offset(torch.tensor([0, 1, 0], dtype=p.dtype, device=p.device))
        n110 = get_hash_offset(torch.tensor([1, 1, 0], dtype=p.dtype, device=p.device))
        n001 = get_hash_offset(torch.tensor([0, 0, 1], dtype=p.dtype, device=p.device))
        n101 = get_hash_offset(torch.tensor([1, 0, 1], dtype=p.dtype, device=p.device))
        n011 = get_hash_offset(torch.tensor([0, 1, 1], dtype=p.dtype, device=p.device))
        n111 = get_hash_offset(torch.tensor([1, 1, 1], dtype=p.dtype, device=p.device))

        x0 = GSEffects.mix(n000, n100, f[:, 0:1])
        x1 = GSEffects.mix(n010, n110, f[:, 0:1])
        x2 = GSEffects.mix(n001, n101, f[:, 0:1])
        x3 = GSEffects.mix(n011, n111, f[:, 0:1])

        y0 = GSEffects.mix(x0, x1, f[:, 1:2])
        y1 = GSEffects.mix(x2, x3, f[:, 1:2])

        return GSEffects.mix(y0, y1, f[:, 2:3])

    @staticmethod
    def rot_2d(angle):
        if isinstance(angle, torch.Tensor):
            s = torch.sin(angle)
            c = torch.cos(angle)
            rot = torch.stack([torch.stack([c, -s], dim=-1),
                                torch.stack([s, c], dim=-1)], dim=-2).squeeze()
        else:
            s = np.sin(angle)
            c = np.cos(angle)
            rot = torch.tensor([[c, -s], [s, c]]).float()
        return rot

    def twister(self, pos, scale, t):
        """Ported verbatim; NOT reachable via apply_effect() in the official source -- see module docstring."""
        h = self.hash(pos)[:, 0:1] + 0.1
        pos_xz_len = self.length_xz(pos)
        s = self.smoothstep(0.0, 8.0, t * t * 0.1 - pos_xz_len * 2.0 + 2.0)[:, None]
        mask = (torch.linalg.norm(scale, dim=-1, keepdim=True) < 0.05)
        pos_y = torch.where(mask, (-10. + pos[:, 1:2]) * (s ** (2 * h)), pos[:, 1:2])
        pos_xz = pos[:, [0, 2]] * torch.exp(-1 * torch.linalg.norm(pos[:, [0, 2]], dim=-1, keepdim=True))
        pos_xz = torch.einsum("n i, n i j -> n j", pos_xz, self.rot_2d(t * 0.2 + pos[:, 1:2] * 20. * (1 - s)))
        pos_new = torch.cat([pos_xz[:, 0:1], pos_y, pos_xz[:, 1:2]], dim=-1)
        return pos_new, s ** 4

    def rain(self, pos, scale, t):
        """Ported verbatim; NOT reachable via apply_effect() in the official source -- see module docstring."""
        h = self.hash(pos)
        pos_xz_len = self.length_xz(pos)
        s = self.smoothstep(0.0, 5.0, t * t * 0.1 - pos_xz_len * 2.0 + 1.0) ** (0.5 + h[:, 0])
        y = pos[:, 1:2]
        pos_y = torch.minimum(-10. + s[:, None] * 15., pos[:, 1:2])
        pos_x = pos[:, 0:1] + pos_y * 0.2
        pos_xz = torch.cat([pos_x, pos[:, 2:3]], dim=-1)
        pos_xz = pos_xz * torch.matmul(self.rot_2d(t * 0.3), torch.ones_like(pos_xz).unsqueeze(-1)).squeeze(-1)
        pos_new = torch.cat([pos_xz[:, 0:1], pos_y, pos_xz[:, 1:2]], dim=-1)
        a = self.smoothstep(-10.0, y.squeeze(), pos_y.squeeze())[:, None]
        return pos_new, a

    def apply_effect(self, gsplat, t, effect_type, ignore_scale=False):
        """
        Args:
            gsplat: dict with 'means' (n,3), 'scales' (n,3), 'colors' (n,3), 'quats' (n,4), 'opacities' (n,)
            t: current time (normalized against start_time/end_time)
            effect_type: only 2 ("Spread") is implemented -- matches official source exactly

        Returns:
            (modified gsplat dict, smoothstep_val or None)
        """
        normalized_t = t - self.start_time
        device = gsplat['means'].device
        dtype = gsplat['means'].dtype

        output = {
            'means': gsplat['means'].clone(),
            'quats': gsplat['quats'].clone(),
            'scales': gsplat['scales'].clone(),
            'opacities': gsplat['opacities'].clone(),
            'colors': gsplat['colors'].clone()
        }

        s = self.smoothstep(0.0, 10.0, normalized_t - 3.2) * 10.0
        scales = output['scales']
        local_pos = output['means'].clone()
        l = self.length_xz(local_pos)
        smoothstep_val = None

        if effect_type == 2:  # Spread Effect
            border = torch.abs(s - l - 0.5)
            decay = 1.0 - 0.2 * torch.exp(-20.0 * border)
            local_pos = local_pos * decay[:, None]

            smoothstep_val = self.smoothstep(s - 0.5, s, l + 0.5)
            if not ignore_scale:
                final_scales = self.mix(scales, 1e-9, smoothstep_val[:, None])
            else:
                final_scales = scales

            noise_input = torch.stack([
                local_pos[:, 0] * 2.0 + normalized_t * 0.5,
                local_pos[:, 1] * 2.0 + normalized_t * 0.5,
                local_pos[:, 2] * 2.0 + normalized_t * 0.5
            ], dim=1)
            noise_val = self.noise(noise_input)

            output['means'] = local_pos + 0.0 * noise_val * smoothstep_val[:, None]
            output['scales'] = final_scales

            at = torch.atan2(local_pos[:, 0], local_pos[:, 2]) / 3.1416
            output['colors'] *= self.step(at, normalized_t - 3.1416)[:, None]
            output['colors'] += (torch.exp(-20.0 * border) +
                                  torch.exp(-50.0 * torch.abs(normalized_t - at - 3.1416)) * 0.5)[:, None]
            output['opacities'] *= self.step(at, normalized_t - 3.1416)
            output['opacities'] += (torch.exp(-20.0 * border) +
                                     torch.exp(-50.0 * torch.abs(normalized_t - at - 3.1416)) * 0.5)

            mask_prob = smoothstep_val.squeeze() if smoothstep_val.dim() > 1 else smoothstep_val
            if not hasattr(self, "random_vals"):
                self.random_vals = torch.rand(mask_prob.shape, device=device, dtype=dtype)
            mask = self.random_vals < mask_prob * 0.8

            if not ignore_scale:
                output['means'][mask] *= 0
                output['scales'][mask] *= 0
                output['opacities'][mask] *= 0

        return output, smoothstep_val
