"""Depth colorization helper, ported from Tencent's official demo
(src/utils/color_map.py) -- only apply_color_map/apply_color_map_to_image,
the 2D CIELab variant needing `colorspacious` was dropped as unused here."""

import torch
from matplotlib import cm


def apply_color_map(x: torch.Tensor, color_map: str = "inferno") -> torch.Tensor:
    cmap = cm.get_cmap(color_map)
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(image: torch.Tensor, color_map: str = "inferno") -> torch.Tensor:
    """image: (..., H, W) -> (..., 3, H, W)"""
    mapped = apply_color_map(image, color_map)  # (..., H, W, 3)
    return mapped.movedim(-1, -3)
