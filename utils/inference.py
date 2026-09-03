"""
Model loading, caching, and inference utilities.

Provides efficient model management and inference wrappers for HunyuanWorld-Mirror.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

# Ensure the custom node directory is FIRST in Python path for model imports
# This is critical because other custom nodes pollute sys.path
_current_file = Path(__file__).resolve()
_custom_node_dir = _current_file.parent.parent  # Go up from utils/ to custom node root
_custom_node_str = str(_custom_node_dir)

# Remove any existing instance and insert at position 0
while _custom_node_str in sys.path:
    sys.path.remove(_custom_node_str)
sys.path.insert(0, _custom_node_str)

from .memory import MemoryManager


class ModelCache:
    """Thread-safe model cache for ComfyUI."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """
        Get model from cache.

        Args:
            key: Cache key (typically model_name + device + precision)

        Returns:
            Cached model or None if not found
        """
        return cls._cache.get(key, None)

    @classmethod
    def set(cls, key: str, model: Any) -> None:
        """
        Store model in cache.

        Args:
            key: Cache key
            model: Model instance to cache
        """
        cls._cache[key] = model
        print(f"Model cached with key: {key}")

    @classmethod
    def clear(cls, key: Optional[str] = None) -> None:
        """
        Clear model from cache.

        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is None:
            cls._cache.clear()
            print("Model cache cleared")
        elif key in cls._cache:
            del cls._cache[key]
            print(f"Model removed from cache: {key}")

    @classmethod
    def get_size(cls) -> int:
        """Get number of cached models."""
        return len(cls._cache)

    @classmethod
    def list_keys(cls) -> list:
        """Get list of all cache keys."""
        return list(cls._cache.keys())


class InferenceWrapper:
    """Wrapper for HunyuanWorld-Mirror model inference."""

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        precision: str = "fp32"
    ):
        """
        Initialize inference wrapper.

        Args:
            model: HunyuanWorld-Mirror model instance
            device: Device to run on ('cuda' or 'cpu')
            precision: Precision mode ('fp32', 'fp16', 'bf16')
        """
        self.model = model
        self.device = device
        self.precision = precision

        # Set model to eval mode
        self.model.eval()

    @torch.no_grad()
    def infer(
        self,
        images: torch.Tensor,
        condition: Optional[list] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on image sequence.

        Args:
            images: Input images in HWM format [1, N, 3, H, W]
            condition: Optional condition flags [pose, depth, intrinsic]
            **kwargs: Additional model arguments

        Returns:
            Dictionary of output tensors:
                - pts3d: 3D points [1, N, H, W, 3]
                - depth: Depth maps [1, N, H, W]
                - normals: Surface normals [1, N, H, W, 3]
                - conf: Confidence maps [1, N, H, W]
                - camera_poses: Camera poses [1, N, 4, 4]
                - camera_intrinsics: Intrinsics [1, N, 3, 3]
                - gaussian_params: Gaussian splatting parameters (dict)
        """
        # Move images to device
        images = images.to(self.device)

        # Apply precision conversion
        if self.precision == "fp16":
            images = images.half()
        elif self.precision == "bf16":
            images = images.bfloat16()

        # Run inference
        try:
            # Prepare views dict as expected by WorldMirror
            views = {'img': images}

            # Convert condition to cond_flags format
            # condition is [pose, depth, intrinsic], default to [0, 0, 0]
            cond_flags = condition if condition is not None else [0, 0, 0]

            # Call model with correct signature
            outputs = self.model(views, cond_flags=cond_flags, is_inference=True)
        except Exception as e:
            print(f"Inference error: {e}")
            raise

        # Convert outputs back to fp32 for compatibility
        if self.precision in ["fp16", "bf16"]:
            outputs = {k: v.float() if isinstance(v, torch.Tensor) else v
                      for k, v in outputs.items()}

        return outputs

    def clear_memory(self) -> None:
        """Clear GPU memory after inference."""
        MemoryManager.clear_cache()

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get current memory statistics."""
        return MemoryManager.get_memory_stats()


def load_model(
    model_name: str = "tencent/HY-World-2.0",
    device: str = "auto",
    precision: str = "fp32",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    subfolder: str = "HY-WorldMirror-2.0",
) -> Tuple[Any, str]:
    """
    Load WorldMirror 2.0 with caching.

    CONFIRMED LIVE (2026-09-03): swapped from WorldMirror 1.0 (raw safetensors
    state_dict + a from-scratch "just call WorldMirror()" instantiation, which
    silently used constructor DEFAULTS rather than the checkpoint's own config
    -- v1's defaults happened to be close enough to work, but WorldMirror 2.0's
    config.json sets fixed_patch_embed=true against a constructor default of
    False, plus several new kwargs (enable_depth_mask, condition_strategy)
    that don't exist on v1 at all. Bare `WorldMirror()` would silently build
    the WRONG architecture and either fail to load the state dict or load it
    onto mismatched layers. This loader instead follows Tencent's own
    HY-World-2.0 reference implementation (hyworld2/worldrecon/pipeline.py,
    WorldMirrorPipeline.from_pretrained) exactly: resolve a directory holding
    config.json + model.safetensors, build WorldMirror(**config), then load
    weights with a selective (shape-checked) state dict merge rather than a
    strict load.

    Args:
        model_name: HuggingFace repo id or local path. Model files are
            expected under {model_name}/{subfolder}/ (matching HY-World-2.0's
            own multi-model repo layout), OR directly in {model_name} if it
            already contains config.json + model.safetensors (local
            concept_mapping staging, or a bare v2 checkpoint dir).
        device: Target device ('auto', 'cuda', 'cpu')
        precision: Precision mode ('fp32', 'fp16', 'bf16')
        cache_dir: Custom cache directory for HuggingFace downloads
        use_cache: Whether to use model cache
        subfolder: Subfolder inside the repo holding the WorldMirror
            checkpoint. Default matches HY-World-2.0's own repo layout.

    Returns:
        Tuple of (model, cache_key)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = _resolve_wm2_model_dir(model_name, subfolder, cache_dir)
    cache_key = f"{model_dir}_{device}_{precision}"

    if use_cache:
        cached_model = ModelCache.get(cache_key)
        if cached_model is not None:
            print(f"✓ Model loaded from cache: {model_dir}")
            return cached_model, cache_key

    print(f"Loading WorldMirror 2.0 from: {model_dir}")
    print(f"Device: {device}, Precision: {precision}")

    try:
        model = _load_worldmirror2(model_dir, device, precision)
        model.eval()

        if use_cache:
            ModelCache.set(cache_key, model)

        print(f"✓ Model loaded successfully")
        MemoryManager.print_memory_stats("After model loading - ")

        return model, cache_key

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def _has_model_files(path: str) -> bool:
    """A directory holds a real WorldMirror 2.0 checkpoint if it has both
    model.safetensors and a config (config.json, matching HY-World-2.0's own
    HF repo; config.yaml also accepted for parity with the upstream loader,
    though HY-World-2.0's own HF repo only ships config.json)."""
    has_weights = os.path.isfile(os.path.join(path, "model.safetensors"))
    has_config = (os.path.isfile(os.path.join(path, "config.yaml"))
                  or os.path.isfile(os.path.join(path, "config.json")))
    return has_weights and has_config


def _resolve_wm2_model_dir(model_name: str, subfolder: str, cache_dir: Optional[str]) -> str:
    """
    Resolve model_name to a local directory containing config.json + model.safetensors.

    Resolution order (matching Tencent's own _resolve_model_dir in
    hyworld2/worldrecon/pipeline.py, plus a local-ComfyUI-models candidate
    first so concept_mapping pre-staging is honored):
      1. {ComfyUI}/models/HunyuanWorld-Mirror/{subfolder}/ -- concept_mapping's
         pre-staged destination, checked first so a job with a pre-downloaded
         checkpoint never touches the network.
      2. {model_name}/{subfolder} -- local repo root with subfolder (or
         model_name itself, if it's already a real local directory).
      3. {model_name} directly, if it already holds config+weights (bare
         local checkpoint dir, no subfolder nesting).
      4. HuggingFace Hub download via snapshot_download(repo_id=model_name,
         allow_patterns=[f"{subfolder}/*"]) -- best-effort self-heal if
         concept_mapping's pre-stage isn't present on this job (confirmed
         platform limitation: concept_mapping isn't guaranteed across
         Graydient machine classes).
    """
    current_dir = Path(__file__).parent.parent  # ComfyUI-HunyuanWorld-Mirror directory
    if current_dir.parent.name == "custom_nodes":
        comfy_root = current_dir.parent.parent
        local_candidate = comfy_root / "models" / "HunyuanWorld-Mirror" / subfolder
        if local_candidate.is_dir() and _has_model_files(str(local_candidate)):
            print(f"[Init] Found local model at {local_candidate}")
            return str(local_candidate)

    candidate = os.path.join(model_name, subfolder)
    if os.path.isdir(candidate) and _has_model_files(candidate):
        print(f"[Init] Found local model at {candidate}")
        return candidate

    if os.path.isdir(model_name) and _has_model_files(model_name):
        print(f"[Init] Found local model at {model_name}")
        return model_name

    print(f"[Init] Downloading from HuggingFace: {model_name} (subfolder={subfolder})")
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
    from huggingface_hub import snapshot_download
    repo_root = snapshot_download(repo_id=model_name, allow_patterns=[f"{subfolder}/*"])
    resolved = os.path.join(repo_root, subfolder)
    if not _has_model_files(resolved):
        raise FileNotFoundError(
            f"Downloaded repo '{model_name}' but subfolder '{subfolder}' does not "
            f"contain model.safetensors + config. Check the repo/subfolder name."
        )
    return resolved


def _load_model_config(model_dir: str) -> dict:
    """Load WorldMirror constructor kwargs from config.json (or config.yaml,
    for parity with the upstream loader's local-training-checkpoint path)."""
    json_path = os.path.join(model_dir, "config.json")
    yaml_path = os.path.join(model_dir, "config.yaml")
    if os.path.isfile(json_path):
        import json as _json
        with open(json_path) as f:
            return _json.load(f)
    elif os.path.isfile(yaml_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(yaml_path)
        model_cfg = cfg.wrapper.model if hasattr(cfg, "wrapper") else cfg.model
        out = OmegaConf.to_container(model_cfg, resolve=True)
        out.pop("_target_", None)
        return out
    raise FileNotFoundError(f"No config.json or config.yaml in {model_dir}")


def _load_state_dict_selective(model, ckpt_state: dict, source_name: str = "checkpoint") -> None:
    """Merge only shape-matching keys from ckpt_state into model, then strict-load
    the merged result -- exactly Tencent's own _load_state_dict_selective. A plain
    strict load would fail outright on any key WorldMirror 2.0 doesn't (yet) use;
    this mirrors what every param that DOES match actually receives, and reports
    how many keys matched so a silent architecture mismatch doesn't go unnoticed."""
    current = model.state_dict()
    for key in current:
        if key in ckpt_state and current[key].shape == ckpt_state[key].shape:
            current[key] = ckpt_state[key]
    model.load_state_dict(current, strict=True)
    matched = sum(1 for k in current if k in ckpt_state and current[k].shape == ckpt_state[k].shape)
    print(f"  Loaded {matched}/{len(current)} keys from {source_name}")


def _load_worldmirror2(model_dir: str, device: str, precision: str) -> Any:
    """Instantiate WorldMirror(**config) and load model.safetensors selectively --
    the checkpoint's own config.json, not constructor defaults, must drive
    architecture (see load_model()'s docstring for why that distinction matters)."""
    import contextlib
    from safetensors.torch import load_file as load_safetensors

    @contextlib.contextmanager
    def _ensure_path_priority():
        _node_str = str(_custom_node_dir)
        original_path = sys.path.copy()
        for key in [k for k in sys.modules if k.startswith('src.')] + (['src'] if 'src' in sys.modules else []):
            del sys.modules[key]
        while _node_str in sys.path:
            sys.path.remove(_node_str)
        sys.path.insert(0, _node_str)
        try:
            yield
        finally:
            sys.path[:] = original_path

    with _ensure_path_priority():
        from src.models.models.worldmirror import WorldMirror

    model_cfg = _load_model_config(model_dir)
    model = WorldMirror(**model_cfg)

    state = load_safetensors(os.path.join(model_dir, "model.safetensors"))
    _load_state_dict_selective(model, state, source_name=model_dir)
    del state

    model = model.to(device)
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    return model


def prepare_model_inputs(
    images: torch.Tensor,
    camera_poses: Optional[torch.Tensor] = None,
    depth_maps: Optional[torch.Tensor] = None,
    intrinsics: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[list], Dict[str, Any]]:
    """
    Prepare inputs for model inference.

    Args:
        images: Input images [1, N, 3, H, W]
        camera_poses: Optional camera poses [N, 4, 4]
        depth_maps: Optional depth priors [N, H, W]
        intrinsics: Optional camera intrinsics [N, 3, 3]

    Returns:
        Tuple of (images, condition_flags, additional_inputs)
    """
    # Determine condition flags
    condition = None
    if camera_poses is not None or depth_maps is not None or intrinsics is not None:
        condition = [
            camera_poses is not None,  # pose condition
            depth_maps is not None,     # depth condition
            intrinsics is not None      # intrinsic condition
        ]

    # Prepare additional inputs
    additional_inputs = {}
    if camera_poses is not None:
        additional_inputs['camera_poses'] = camera_poses
    if depth_maps is not None:
        additional_inputs['depth_priors'] = depth_maps
    if intrinsics is not None:
        additional_inputs['intrinsics'] = intrinsics

    return images, condition, additional_inputs
