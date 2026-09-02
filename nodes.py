"""
HunyuanWorld-Mirror ComfyUI Node Pack - Main Nodes

All 8 core nodes for 3D reconstruction from images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Dict, Any, Optional

from .utils import (
    comfy_to_hwm,
    hwm_to_comfy,
    normalize_depth,
    normals_to_rgb,
    MemoryManager,
    ModelCache,
    InferenceWrapper,
    ExportUtils,
    tensor_to_numpy,
)
from .utils.inference import load_model
from .utils.edge_mask import depth_edge, normals_edge
from .utils.sky_segmentation import download_skyseg_model, segment_sky_mask
from .utils.flythrough import render_gaussian_flythrough
from .utils.scene_composite import build_composite_scene


# ============================================================================
# Node 0: PreprocessImagesForHWM (NEW in Phase 2)
# ============================================================================

class PreprocessImagesForHWM:
    """
    Preprocess images for HunyuanWorld-Mirror model with professional crop/pad strategies.

    Ensures images meet model requirements:
    - Dimensions divisible by 14 (patch size)
    - Consistent sizing across batch
    - Proper aspect ratio handling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to preprocess. Can be single or batch of images from LoadImage or other nodes."
                }),
                "strategy": (["crop", "pad"], {
                    "default": "crop",
                    "tooltip": "Preprocessing strategy. CROP: resize width to target, center-crop height if too tall. PAD: scale largest dimension to target, pad smaller dimension with white."
                }),
                "target_size": ("INT", {
                    "default": 518,
                    "min": 224,
                    "max": 1024,
                    "step": 14,
                    "tooltip": "Target image size in pixels (must be divisible by 14). Default 518 is optimal for the model. Larger = more detail but slower inference."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_images",)
    FUNCTION = "preprocess"
    CATEGORY = "HunyuanWorld-Mirror/preprocessing"

    def preprocess(
        self,
        images: torch.Tensor,
        strategy: str,
        target_size: int
    ) -> Tuple[torch.Tensor]:
        """Preprocess images with crop or pad strategy."""

        # ComfyUI images are [B, H, W, C] in range [0, 1]
        batch_size, height, width, channels = images.shape

        print(f"\n{'='*60}")
        print(f"Preprocessing {batch_size} images")
        print(f"Input size: {width}x{height}, Target: {target_size}x{target_size}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        # Ensure target_size is divisible by 14
        target_size = (target_size // 14) * 14

        processed_images = []

        for i in range(batch_size):
            # Get single image [H, W, C]
            img = images[i]

            # Convert to CHW format for processing
            img_chw = img.permute(2, 0, 1)  # [C, H, W]

            if strategy == "crop":
                # Resize width to target, maintain aspect ratio for height
                aspect_ratio = height / width
                new_width = target_size
                new_height = int(new_width * aspect_ratio)
                # Round to nearest multiple of 14
                new_height = (new_height // 14) * 14

                # Resize
                img_resized = torch.nn.functional.interpolate(
                    img_chw.unsqueeze(0),
                    size=(new_height, new_width),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)

                # Center crop if height > target_size
                if new_height > target_size:
                    crop_start = (new_height - target_size) // 2
                    img_final = img_resized[:, crop_start:crop_start + target_size, :]
                else:
                    # Pad if height < target_size
                    pad_needed = target_size - new_height
                    pad_top = pad_needed // 2
                    pad_bottom = pad_needed - pad_top
                    img_final = torch.nn.functional.pad(
                        img_resized,
                        (0, 0, pad_top, pad_bottom),
                        mode='constant',
                        value=1.0  # White padding
                    )

            else:  # strategy == "pad"
                # Scale largest dimension to target, pad smaller
                if width >= height:
                    new_width = target_size
                    new_height = int(height * (target_size / width))
                    new_height = (new_height // 14) * 14
                else:
                    new_height = target_size
                    new_width = int(width * (target_size / height))
                    new_width = (new_width // 14) * 14

                # Resize
                img_resized = torch.nn.functional.interpolate(
                    img_chw.unsqueeze(0),
                    size=(new_height, new_width),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)

                # Pad to square
                pad_height = target_size - new_height
                pad_width = target_size - new_width

                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left

                img_final = torch.nn.functional.pad(
                    img_resized,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=1.0  # White padding
                )

            # Convert back to HWC format
            img_final = img_final.permute(1, 2, 0)  # [H, W, C]
            processed_images.append(img_final)

        # Stack back to batch
        output = torch.stack(processed_images, dim=0)

        print(f"✓ Preprocessed to {output.shape[2]}x{output.shape[1]}")
        print(f"{'='*60}\n")

        return (output,)


# ============================================================================
# Node 1: LoadHunyuanWorldMirrorModel
# ============================================================================

class LoadHunyuanWorldMirrorModel:
    """
    Load the HunyuanWorld-Mirror model with automatic caching.

    This node loads the model once and caches it for reuse across workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "HunyuanWorld-Mirror",
                    "multiline": False,
                    "tooltip": "Model name, filename, or path. Checks ComfyUI/models/HunyuanWorld-Mirror/ first. Examples: 'HunyuanWorld-Mirror', 'model.safetensors', or full path."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Which device to run the model on. 'auto' selects CUDA if available, otherwise CPU. Use 'cuda' for GPU acceleration (recommended) or 'cpu' for compatibility."
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "bf16",
                    "tooltip": "Numeric precision for model weights. bf16 (bfloat16) matches Tencent's own official demo (app.py) and has much better dynamic range than fp16 for this model's Gaussian scale/rotation predictions -- fp16 overflow on uncertain/extreme predictions is a plausible source of streaky splat artifacts. fp32 is more accurate still but uses 2x the memory of bf16. Falls back to fp32 automatically on GPUs without bf16 support."
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload the model from disk, bypassing cache. Use this if you updated the model files or if the model isn't working correctly."
                }),
            },
            "optional": {
                "cache_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom directory for downloading and caching model files from HuggingFace. Leave empty to use default cache location."
                }),
            }
        }

    RETURN_TYPES = ("HWMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanWorld-Mirror/loaders"

    def load_model(
        self,
        model_name: str,
        device: str,
        precision: str,
        force_reload: bool,
        cache_dir: str = ""
    ) -> Tuple[Any]:
        """Load and cache the model."""

        print("=" * 60)
        print("Loading HunyuanWorld-Mirror Model")
        print("=" * 60)

        try:
            # Load model (with optional cache bypass)
            model, cache_key = load_model(
                model_name=model_name,
                device=device,
                precision=precision,
                cache_dir=cache_dir if cache_dir else None,
                use_cache=not force_reload  # Bypass cache if force_reload is True
            )

            if force_reload:
                print("* Model reloaded from disk (cache bypassed)")

            print(f"Model ready: {cache_key}")
            print("=" * 60)

            return (model,)

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise


# ============================================================================
# Node 2: HWMInference
# ============================================================================

class HWMInference:
    """
    Main inference node - generates all 3D outputs in a single pass.

    Outputs: depth, normals, points3d, camera_poses, camera_intrinsics, gaussians
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HWMIRROR_MODEL", {
                    "tooltip": "The loaded HunyuanWorld-Mirror model from the 'Load HunyuanWorld-Mirror Model' node."
                }),
                "images": ("IMAGE", {
                    "tooltip": "Sequence of input images (4-64 frames). Use LoadImage + ImageBatch to create a sequence. More frames give better 3D reconstruction but use more memory."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducible results. Set to -1 for random seed each time, or use a specific number (e.g., 42) to get the same results every run."
                }),
                "batch_size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Process images in batches of this size to avoid OOM errors. Lower values use less memory but take longer. Recommended: 8-16 for large sequences (50+ images), 32+ for smaller sequences."
                }),
            },
        }

    RETURN_TYPES = ("DEPTH", "NORMALS", "POINTS3D", "POSES", "INTRINSICS", "GAUSSIANS", "CONFIDENCE", "CONFIDENCE", "CONFIDENCE")
    RETURN_NAMES = ("depth", "normals", "points3d", "camera_poses", "camera_intrinsics", "gaussians", "depth_conf", "normals_conf", "pts3d_conf")
    FUNCTION = "infer"
    CATEGORY = "HunyuanWorld-Mirror/inference"

    def infer(
        self,
        model: Any,
        images: torch.Tensor,
        seed: int,
        batch_size: int
    ) -> Tuple:
        """Run inference on image sequence with batching support."""

        print("\n" + "=" * 60)
        print("HunyuanWorld-Mirror Inference")
        print("=" * 60)

        # Set seed if provided
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Get input info
        B, H, W, C = images.shape
        print(f"Input: {B} images at {H}x{W}x{C}")

        # Get device info
        device = next(model.parameters()).device
        precision = "fp16" if next(model.parameters()).dtype == torch.float16 else "fp32"

        # Determine number of batches
        num_batches = (B + batch_size - 1) // batch_size

        if num_batches > 1:
            print(f"Processing in {num_batches} batches of {batch_size} images each")

        # Estimate memory per batch
        estimated_mem = MemoryManager.estimate_sequence_memory(min(B, batch_size), H, W, precision)
        print(f"Estimated memory per batch: {estimated_mem:.2f}GB")

        # Check memory
        available, msg = MemoryManager.check_memory_available(estimated_mem)
        if not available:
            print(f"Warning: {msg}")

        try:
            # Create inference wrapper
            wrapper = InferenceWrapper(model, str(device), precision)

            # Process batches
            all_depth = []
            all_normals = []
            all_pts3d = []
            all_poses = []
            all_intrinsics = []
            all_gaussian_means = []
            all_gaussian_scales = []
            all_gaussian_quats = []
            all_gaussian_colors = []
            all_gaussian_sh = []
            all_gaussian_opacities = []
            # Confidence maps
            all_depth_conf = []
            all_normals_conf = []
            all_pts3d_conf = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, B)
                batch_images = images[start_idx:end_idx]

                if num_batches > 1:
                    print(f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx-1})...")

                # Convert to HWM format
                hwm_images = comfy_to_hwm(batch_images)  # [B, H, W, C] -> [1, N, 3, H, W]

                # Run inference
                MemoryManager.print_memory_stats("  GPU Memory - ")

                with torch.no_grad():
                    outputs = wrapper.infer(hwm_images, condition=None)

                # Debug: Log available output keys
                if batch_idx == 0:
                    output_keys = list(outputs.keys())
                    print(f"  Model outputs: {', '.join(output_keys)}")

                # Extract and collect outputs
                depth = outputs.get('depth', outputs.get('pred_depth', None))
                normals = outputs.get('normals', outputs.get('pred_normals', None))
                pts3d = outputs.get('pts3d', outputs.get('pred_pts3d', None))
                poses = outputs.get('camera_poses', outputs.get('pred_poses', None))
                intrinsics = outputs.get('camera_intrinsics', outputs.get('camera_intrs', None))

                # Extract confidence maps
                depth_conf = outputs.get('depth_conf', outputs.get('pred_depth_conf', None))
                normals_conf = outputs.get('normals_conf', outputs.get('pred_normals_conf', None))
                pts3d_conf = outputs.get('pts3d_conf', outputs.get('pred_pts3d_conf', None))

                if depth is not None:
                    all_depth.append(depth)
                if normals is not None:
                    all_normals.append(normals)
                if pts3d is not None:
                    all_pts3d.append(pts3d)
                if poses is not None:
                    all_poses.append(poses)
                if intrinsics is not None:
                    all_intrinsics.append(intrinsics)

                # Collect confidence maps
                if depth_conf is not None:
                    all_depth_conf.append(depth_conf)
                if normals_conf is not None:
                    all_normals_conf.append(normals_conf)
                if pts3d_conf is not None:
                    all_pts3d_conf.append(pts3d_conf)

                # Extract Gaussian parameters from splats dictionary
                splats = outputs.get('splats', None)
                if splats is not None and isinstance(splats, dict):
                    # Debug: log splats structure in first batch
                    if batch_idx == 0 and num_batches > 1:
                        print(f"  Splats structure: {', '.join([f'{k}: {type(v).__name__}' for k, v in splats.items()])}")

                    # Extract individual Gaussian parameters
                    # Handle both tensor and list values
                    def extract_param(key):
                        """Extract parameter, converting list to tensor if needed."""
                        if key not in splats or splats[key] is None:
                            return None
                        val = splats[key]
                        # If it's a list, convert to tensor
                        if isinstance(val, list):
                            if len(val) == 0:
                                return None
                            # If list contains tensors, stack them
                            if isinstance(val[0], torch.Tensor):
                                return torch.stack(val)
                            # Otherwise convert the list directly
                            return torch.tensor(val)
                        return val

                    means = extract_param('means')
                    if means is not None:
                        all_gaussian_means.append(means)

                    scales = extract_param('scales')
                    if scales is not None:
                        all_gaussian_scales.append(scales)

                    quats = extract_param('quats')
                    if quats is not None:
                        all_gaussian_quats.append(quats)

                    opacities = extract_param('opacities')
                    if opacities is not None:
                        all_gaussian_opacities.append(opacities)

                    # Extract colors and spherical harmonics separately
                    sh = extract_param('sh')
                    if sh is not None:
                        all_gaussian_sh.append(sh)

                    colors = extract_param('colors')
                    if colors is not None:
                        all_gaussian_colors.append(colors)

                # Clear batch memory
                if num_batches > 1:
                    MemoryManager.clear_cache()

            # Concatenate results along the batch dimension
            print("Concatenating batch results...")

            def concat_tensors(tensor_list, dim=0):
                """Concatenate tensors along specified dimension if list is not empty."""
                if len(tensor_list) == 0:
                    return None
                if len(tensor_list) == 1:
                    return tensor_list[0]

                # Ensure all items are tensors
                tensors = []
                for item in tensor_list:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item)
                    elif isinstance(item, list):
                        # Convert list to tensor
                        if len(item) > 0 and isinstance(item[0], torch.Tensor):
                            tensors.append(torch.stack(item))
                        else:
                            tensors.append(torch.tensor(item))
                    else:
                        print(f"Warning: Skipping non-tensor item of type {type(item)}")
                        continue

                if len(tensors) == 0:
                    return None
                if len(tensors) == 1:
                    return tensors[0]

                # Handle different tensor shapes - some may have [1, N, ...] format
                first_shape = tensors[0].shape
                if len(first_shape) > 1 and first_shape[0] == 1:
                    # Concatenate along dimension 1 (the N dimension in [1, N, ...])
                    return torch.cat(tensors, dim=1)
                else:
                    # Concatenate along dimension 0 (batch dimension)
                    return torch.cat(tensors, dim=dim)

            depth = concat_tensors(all_depth)
            normals = concat_tensors(all_normals)
            pts3d = concat_tensors(all_pts3d)
            poses = concat_tensors(all_poses)
            intrinsics = concat_tensors(all_intrinsics)

            # Concatenate confidence maps
            depth_conf = concat_tensors(all_depth_conf)
            normals_conf = concat_tensors(all_normals_conf)
            pts3d_conf = concat_tensors(all_pts3d_conf)

            # Concatenate Gaussian parameters
            gaussian_params = {
                'means': concat_tensors(all_gaussian_means),
                'scales': concat_tensors(all_gaussian_scales),
                'quats': concat_tensors(all_gaussian_quats),
                'colors': concat_tensors(all_gaussian_colors),
                'opacities': concat_tensors(all_gaussian_opacities),
                'sh': concat_tensors(all_gaussian_sh) if len(all_gaussian_sh) > 0 else None,
            }

            # Log Gaussian availability
            if gaussian_params['means'] is not None:
                print(f"  Gaussians: Generated successfully")
            else:
                print(f"  Gaussians: Not available (model may need enable_gs=True or multiple views)")

            print("✓ Inference complete")
            print(f"  Depth: {depth.shape if depth is not None else 'N/A'}")
            print(f"  Normals: {normals.shape if normals is not None else 'N/A'}")
            print(f"  Points3D: {pts3d.shape if pts3d is not None else 'N/A'}")
            print(f"  Poses: {poses.shape if poses is not None else 'N/A'}")
            print(f"  Intrinsics: {intrinsics.shape if intrinsics is not None else 'N/A'}")
            if depth_conf is not None:
                print(f"  Confidence Maps: Available")
            print("=" * 60)

            # Clear memory
            wrapper.clear_memory()

            return (depth, normals, pts3d, poses, intrinsics, gaussian_params, depth_conf, normals_conf, pts3d_conf)

        except Exception as e:
            print(f"✗ Inference error: {e}")
            MemoryManager.clear_cache()
            raise


# ============================================================================
# Node 3: VisualizeDepth
# ============================================================================

class VisualizeDepth:
    """
    Convert depth maps to colorized visualizations for preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("DEPTH", {
                    "tooltip": "Depth map output from the HWM Inference node. Contains distance information for each pixel in the image."
                }),
                "colormap": (["viridis", "plasma", "turbo", "magma", "inferno", "gray"], {
                    "default": "turbo",
                    "tooltip": "Color scheme for visualizing depth. 'turbo' is rainbow-like (blue=close, red=far), 'viridis' is blue-to-yellow, 'gray' is grayscale. Choose based on preference."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to automatically adjust depth values to use the full color range. Enable (True) for better visualization, disable (False) to use raw depth values."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "HunyuanWorld-Mirror/visualization"

    def visualize(
        self,
        depth: torch.Tensor,
        colormap: str,
        normalize: bool
    ) -> Tuple[torch.Tensor]:
        """Convert depth to colored image."""

        # Convert to numpy
        depth_np = tensor_to_numpy(depth)

        # Handle batch dimension
        if depth_np.ndim == 3:
            # [N, H, W]
            batch_size = depth_np.shape[0]
        elif depth_np.ndim == 4:
            # [1, N, H, W]
            depth_np = depth_np.squeeze(0)
            batch_size = depth_np.shape[0]
        elif depth_np.ndim == 5:
            # [1, N, H, W, 1] - squeeze first and last dimensions
            depth_np = depth_np.squeeze(0).squeeze(-1)
            batch_size = depth_np.shape[0]
        else:
            raise ValueError(f"Unexpected depth shape: {depth_np.shape}")

        # Process each depth map
        colored_images = []
        cmap = cm.get_cmap(colormap)

        for i in range(batch_size):
            depth_single = depth_np[i]

            # Normalize
            if normalize:
                d_min = depth_single.min()
                d_max = depth_single.max()
                depth_norm = (depth_single - d_min) / (d_max - d_min + 1e-8)
            else:
                depth_norm = depth_single

            # Apply colormap
            colored = cmap(depth_norm)  # Returns RGBA
            colored_rgb = colored[:, :, :3]  # Take RGB only

            colored_images.append(colored_rgb)

        # Stack and convert to torch tensor
        images_np = np.stack(colored_images, axis=0)  # [N, H, W, 3]
        images_torch = torch.from_numpy(images_np).float()

        print(f"✓ Visualized {batch_size} depth maps with '{colormap}' colormap")

        return (images_torch,)


# ============================================================================
# Node 4: VisualizeNormals
# ============================================================================

class VisualizeNormals:
    """
    Convert surface normals to RGB visualization.
    Standard mapping: X→Red, Y→Green, Z→Blue
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normals": ("NORMALS", {
                    "tooltip": "Surface normal vectors from the HWM Inference node. Shows which direction each surface is facing in 3D space. Converted to RGB where X=Red, Y=Green, Z=Blue."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "HunyuanWorld-Mirror/visualization"

    def visualize(self, normals: torch.Tensor) -> Tuple[torch.Tensor]:
        """Convert normals to RGB image."""

        # Handle batch dimension - squeeze if present
        if normals.ndim == 5:
            # [1, N, H, W, 3] -> [N, H, W, 3]
            normals = normals.squeeze(0)

        # Normals are expected in range [-1, 1]
        # Convert to [0, 1] for visualization
        normals_rgb = normals_to_rgb(normals)

        print(f"✓ Visualized normals: {normals_rgb.shape}")

        return (normals_rgb,)


# ============================================================================
# Node 5: SavePointCloud
# ============================================================================

class SavePointCloud:
    """
    Export 3D point cloud to standard formats.
    Supports: PLY, PCD, OBJ, XYZ
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points3d": ("POINTS3D", {
                    "tooltip": "3D point coordinates from the HWM Inference node. Each point represents a location in 3D space (X, Y, Z coordinates)."
                }),
                "filepath": ("STRING", {
                    "default": "./output/pointcloud.ply",
                    "multiline": False,
                    "tooltip": "Where to save the point cloud file. Can be relative (./output/file.ply) or absolute path (C:/Users/Name/Documents/file.ply). File extension will auto-adjust to match format."
                }),
                "format": (["ply", "obj", "xyz"], {
                    "default": "ply",
                    "tooltip": "File format for the point cloud. PLY is most common and supports colors/normals. OBJ works with most 3D software. XYZ is simple text format (just coordinates)."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Filter out low-confidence points. 0=keep all points, 50=keep top 50%, 95=keep only very confident points. Higher values remove more noise but may lose details."
                }),
                "subsample_factor": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Reduce point count by keeping every Nth point. 1=keep all, 2=keep half, 4=keep quarter. Useful for reducing file size and improving viewer performance with large point clouds (10M+ points)."
                }),
            },
            "optional": {
                "colors": ("IMAGE", {
                    "tooltip": "Optional: RGB colors for each point, typically from the source images. Makes the point cloud look more realistic when viewed in 3D software."
                }),
                "normals": ("NORMALS", {
                    "tooltip": "Optional: Surface normal directions for each point. Helps with lighting and rendering in 3D viewers. Only supported in PLY format."
                }),
                "confidence": ("*", {
                    "tooltip": "Optional: Confidence values for each point from HWM Inference (pts3d_conf output). Used with confidence_threshold to filter low-quality points."
                }),
                "depth": ("DEPTH", {
                    "tooltip": "Optional: Depth map from HWM Inference. Required (along with normals) for apply_edge_mask -- removes 'flying pixel' points at depth/normal discontinuities, the same technique Tencent's own official demo (app.py) uses before export."
                }),
                "apply_edge_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove points at depth/normal discontinuities (silhouette edges, where triangulation is unreliable) before export. Matches Tencent's official demo behavior. Requires both depth and normals to be connected."
                }),
                "edge_normal_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 90.0,
                    "step": 0.5,
                    "tooltip": "Normal-angle discontinuity tolerance in degrees (official demo default: 5.0). Lower = more aggressive edge removal."
                }),
                "edge_depth_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.005,
                    "tooltip": "Relative depth discontinuity tolerance (official demo default: 0.03). Lower = more aggressive edge removal."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        points3d: torch.Tensor,
        filepath: str,
        format: str,
        confidence_threshold: float,
        subsample_factor: int,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        apply_edge_mask: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03
    ) -> Tuple[str]:
        """Save point cloud to file with optional confidence filtering and subsampling."""

        # Check if 3D points are available
        if points3d is None:
            print("⚠ Warning: 3D points not available - skipping export")
            return ("",)

        # Convert to numpy
        points_np = tensor_to_numpy(points3d)
        colors_np = tensor_to_numpy(colors) if colors is not None else None
        normals_np = tensor_to_numpy(normals) if normals is not None else None
        confidence_np = tensor_to_numpy(confidence) if confidence is not None else None

        # Fix dimension mismatch: resize colors/normals to match points3d if needed
        if colors_np is not None:
            # Get spatial dimensions
            # points3d: [1, N, H, W, 3] or [N, H, W, 3]
            # colors: [N, H_orig, W_orig, 3]
            points_shape = points_np.shape
            colors_shape = colors_np.shape

            # Determine points spatial dimensions
            if len(points_shape) == 5:
                # [1, N, H, W, 3]
                pts_h, pts_w = points_shape[2], points_shape[3]
            elif len(points_shape) == 4:
                # [N, H, W, 3]
                pts_h, pts_w = points_shape[1], points_shape[2]
            else:
                pts_h, pts_w = None, None

            # Determine colors spatial dimensions
            if len(colors_shape) == 4:
                # [N, H, W, 3]
                col_h, col_w = colors_shape[1], colors_shape[2]
            else:
                col_h, col_w = None, None

            # Resize if dimensions don't match
            if pts_h is not None and col_h is not None and (pts_h != col_h or pts_w != col_w):
                print(f"  Resizing colors from {col_h}x{col_w} to {pts_h}x{pts_w} to match points3d")
                # Use torch for resizing then convert back
                colors_torch = torch.from_numpy(colors_np)
                # [N, H, W, 3] -> [N, 3, H, W] for interpolate
                colors_torch = colors_torch.permute(0, 3, 1, 2)
                colors_torch = torch.nn.functional.interpolate(
                    colors_torch,
                    size=(pts_h, pts_w),
                    mode='bilinear',
                    align_corners=False
                )
                # [N, 3, H, W] -> [N, H, W, 3]
                colors_torch = colors_torch.permute(0, 2, 3, 1)
                colors_np = colors_torch.numpy()

        # Similar fix for normals
        if normals_np is not None:
            points_shape = points_np.shape
            normals_shape = normals_np.shape

            # Determine points spatial dimensions
            if len(points_shape) == 5:
                pts_h, pts_w = points_shape[2], points_shape[3]
            elif len(points_shape) == 4:
                pts_h, pts_w = points_shape[1], points_shape[2]
            else:
                pts_h, pts_w = None, None

            # Determine normals spatial dimensions
            if len(normals_shape) == 5:
                # [1, N, H, W, 3]
                norm_h, norm_w = normals_shape[2], normals_shape[3]
            elif len(normals_shape) == 4:
                # [N, H, W, 3]
                norm_h, norm_w = normals_shape[1], normals_shape[2]
            else:
                norm_h, norm_w = None, None

            # Resize if dimensions don't match
            if pts_h is not None and norm_h is not None and (pts_h != norm_h or pts_w != norm_w):
                print(f"  Resizing normals from {norm_h}x{norm_w} to {pts_h}x{pts_w} to match points3d")
                # Remove batch dim if present
                if len(normals_shape) == 5:
                    normals_np = normals_np.squeeze(0)
                normals_torch = torch.from_numpy(normals_np)
                # [N, H, W, 3] -> [N, 3, H, W]
                normals_torch = normals_torch.permute(0, 3, 1, 2)
                normals_torch = torch.nn.functional.interpolate(
                    normals_torch,
                    size=(pts_h, pts_w),
                    mode='bilinear',
                    align_corners=False
                )
                # [N, 3, H, W] -> [N, H, W, 3]
                normals_torch = normals_torch.permute(0, 2, 3, 1)
                normals_np = normals_torch.numpy()

        # Depth/normal discontinuity ("flying pixel") edge masking, ported from
        # Tencent's own official demo (app.py) -- see utils/edge_mask.py. Applied
        # here (points3d), not to Save3DGaussians: the model's internal prune_gs
        # voxel-merge already reorders/dedupes exported Gaussians away from a
        # per-pixel grid by the time they reach this node pack, so there is no
        # valid per-pixel correspondence left to mask against on the splat path --
        # the official demo doesn't apply this to splats either, only to points3d.
        edge_valid_mask = None
        if apply_edge_mask and depth is not None and normals_np is not None:
            depth_np = tensor_to_numpy(depth)
            # Normalize depth to [num_frames, H, W]
            if depth_np.ndim == 5:
                depth_np = depth_np.squeeze(0)
                if depth_np.shape[-1] == 1:
                    depth_np = depth_np.squeeze(-1)
            elif depth_np.ndim == 4:
                if depth_np.shape[-1] == 1:
                    depth_np = depth_np.squeeze(-1)

            # Normalize normals_np (already resized to match points3d's H,W above) to [num_frames, H, W, 3]
            normals_for_edge = normals_np
            if normals_for_edge.ndim == 5:
                normals_for_edge = normals_for_edge.squeeze(0)

            pts_h, pts_w = points_np.shape[-3], points_np.shape[-2]
            if depth_np.shape[-2:] != (pts_h, pts_w):
                depth_torch = torch.from_numpy(depth_np).unsqueeze(1)  # [N,1,H,W]
                depth_torch = torch.nn.functional.interpolate(depth_torch, size=(pts_h, pts_w), mode='bilinear', align_corners=False)
                depth_np = depth_torch.squeeze(1).numpy()

            if depth_np.shape[0] == normals_for_edge.shape[0]:
                per_frame_masks = []
                for i in range(depth_np.shape[0]):
                    d_edges = depth_edge(depth_np[i], rtol=edge_depth_threshold)
                    n_edges = normals_edge(normals_for_edge[i], tol=edge_normal_threshold)
                    per_frame_masks.append(~(d_edges & n_edges))
                edge_valid_mask = np.stack(per_frame_masks, axis=0)  # [N, H, W]
                print(f"  Edge mask computed: {edge_valid_mask.sum()}/{edge_valid_mask.size} pixels kept (normal_tol={edge_normal_threshold} deg, depth_rtol={edge_depth_threshold})")
            else:
                print(f"  Warning: depth frame count ({depth_np.shape[0]}) != normals frame count ({normals_for_edge.shape[0]}) -- skipping edge mask")

        # Apply subsampling if requested
        if subsample_factor > 1:
            # Flatten all arrays to 1D list of points
            points_flat = points_np.reshape(-1, 3)
            num_original = len(points_flat)

            # Create subsample mask
            subsample_mask = np.arange(num_original) % subsample_factor == 0

            # Apply to all arrays
            points_np = points_flat[subsample_mask].reshape(-1, 3)

            if colors_np is not None:
                colors_flat = colors_np.reshape(-1, 3)
                colors_np = colors_flat[subsample_mask].reshape(-1, 3)

            if normals_np is not None:
                normals_flat = normals_np.reshape(-1, 3)
                normals_np = normals_flat[subsample_mask].reshape(-1, 3)

            if confidence_np is not None:
                confidence_flat = confidence_np.reshape(-1)
                confidence_np = confidence_flat[subsample_mask].reshape(-1)

            if edge_valid_mask is not None:
                edge_flat = edge_valid_mask.reshape(-1)
                edge_valid_mask = edge_flat[subsample_mask]

            print(f"  Subsampling: {subsample_mask.sum()}/{num_original} points (factor={subsample_factor})")

        # Ensure file extension matches format
        if not filepath.endswith(f'.{format}'):
            filepath = filepath.rsplit('.', 1)[0] + f'.{format}'

        # Save based on format
        if format == "ply":
            saved_path = ExportUtils.save_point_cloud_ply(
                filepath, points_np, colors_np, normals_np,
                confidence=confidence_np,
                confidence_threshold=confidence_threshold,
                edge_valid_mask=edge_valid_mask
            )
        elif format == "obj":
            saved_path = ExportUtils.save_point_cloud_obj(
                filepath, points_np, colors_np
            )
        elif format == "xyz":
            saved_path = ExportUtils.save_point_cloud_xyz(
                filepath, points_np, colors_np
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node 6: Save3DGaussians
# ============================================================================

class Save3DGaussians:
    """
    Export 3D Gaussian Splatting representation.
    Standard 3DGS PLY format compatible with viewers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gaussians": ("GAUSSIANS", {
                    "tooltip": "3D Gaussian Splatting parameters from the HWM Inference node. Contains position, scale, rotation, color, and opacity for each Gaussian primitive."
                }),
                "filepath": ("STRING", {
                    "default": "./output/gaussians.ply",
                    "multiline": False,
                    "tooltip": "Where to save the Gaussian Splatting file. Use .ply extension. This file can be loaded in 3DGS viewers for real-time novel view synthesis."
                }),
                "include_sh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to include Spherical Harmonics coefficients for view-dependent appearance. Enable for more realistic lighting effects, disable for smaller files and faster loading."
                }),
                "filter_scale_percentile": ("FLOAT", {
                    "default": 95.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Remove Gaussians with unusually large scales (outliers/artifacts). 95=keep 95% of Gaussians, 90=more aggressive filtering. 0=disable filtering, 100=keep all."
                }),
                "normalize_colors": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply percentile-based color normalization to reduce high contrast artifacts. Leave off to preserve the model's original colors."
                }),
                "subsample_factor": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Reduce Gaussian count by keeping every Nth Gaussian. 1=keep all, 2=keep half, 4=keep quarter. Higher values create smaller files and faster viewer performance."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        gaussians: Dict[str, torch.Tensor],
        filepath: str,
        include_sh: bool,
        filter_scale_percentile: float,
        normalize_colors: bool,
        subsample_factor: int
    ) -> Tuple[str]:
        """Save Gaussian parameters to PLY file with outlier filtering and optional subsampling."""

        # Check if gaussians are available
        if gaussians is None or gaussians.get('means') is None:
            print("⚠ Warning: Gaussian parameters not available - skipping export")
            print("  Note: 3D Gaussian Splatting may require:")
            print("    - Multiple input images (2+ views)")
            print("    - Specific model configuration")
            print("    - Sufficient texture/feature variation")
            print("  Continuing with available outputs (depth, normals, points3D)...")

            # Return placeholder path to indicate skipped
            return ("",)

        # Extract parameters
        means = tensor_to_numpy(gaussians['means'])
        scales = tensor_to_numpy(gaussians['scales'])
        quats = tensor_to_numpy(gaussians['quats'])
        opacities = tensor_to_numpy(gaussians['opacities'])

        # Handle colors: extract from SH DC term if colors is None
        colors = gaussians.get('colors', None)
        if colors is None:
            # Try to extract from SH DC term
            sh_tensor = gaussians.get('sh', None)
            if sh_tensor is not None:
                sh_numpy = tensor_to_numpy(sh_tensor)
                # Extract first 3 coefficients (DC term) as RGB
                if sh_numpy is not None and sh_numpy.shape[-1] >= 3:
                    colors = sh_numpy[..., :3] if len(sh_numpy.shape) > 2 else sh_numpy[:, :3]
                    print(f"  Extracted RGB colors from SH DC term")

        # Convert colors to numpy if it's still a tensor
        if colors is not None and not isinstance(colors, np.ndarray):
            colors = tensor_to_numpy(colors)

        sh = None
        if include_sh and 'sh' in gaussians:
            sh = tensor_to_numpy(gaussians['sh'])

        # Apply subsampling if requested
        if subsample_factor > 1:
            num_original = len(means)
            subsample_mask = np.arange(num_original) % subsample_factor == 0
            means = means[subsample_mask]
            scales = scales[subsample_mask]
            quats = quats[subsample_mask]
            colors = colors[subsample_mask]
            opacities = opacities[subsample_mask]
            if sh is not None:
                sh = sh[subsample_mask]
            print(f"  Subsampling: {subsample_mask.sum()}/{num_original} Gaussians (factor={subsample_factor})")

        # Ensure .ply extension
        if not filepath.endswith('.ply'):
            filepath = filepath.rsplit('.', 1)[0] + '.ply'

        # Save with scale filtering and color normalization
        saved_path = ExportUtils.save_gaussian_ply(
            filepath, means, scales, quats, colors, opacities, sh,
            filter_scale_percentile=filter_scale_percentile,
            normalize_colors=normalize_colors
        )

        return (saved_path,)


# ============================================================================
# Node 7: SaveDepthMap
# ============================================================================

class SaveDepthMap:
    """
    Export depth maps in various precision formats.
    Supports: NPY, EXR, PFM, PNG16
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("DEPTH", {
                    "tooltip": "Depth map from the HWM Inference node. Raw depth values will be saved with full precision."
                }),
                "filepath": ("STRING", {
                    "default": "./output/depth.npy",
                    "multiline": False,
                    "tooltip": "Where to save the depth data. File extension will auto-adjust to match the selected format (e.g., depth.npy, depth.exr, depth.pfm, depth.png)."
                }),
                "format": (["npy", "exr", "pfm", "png16"], {
                    "default": "npy",
                    "tooltip": "File format for depth data. NPY is NumPy binary (full precision, Python-friendly). EXR is OpenEXR (high dynamic range, used in VFX). PFM is Portable Float Map (standard float format). PNG16 is 16-bit PNG (good compatibility)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        depth: torch.Tensor,
        filepath: str,
        format: str
    ) -> Tuple[str]:
        """Save depth map to file."""

        # Convert to numpy
        depth_np = tensor_to_numpy(depth)

        # Ensure correct extension
        if not filepath.endswith(f'.{format}'):
            filepath = filepath.rsplit('.', 1)[0] + f'.{format}'

        # Save based on format
        if format == "npy":
            saved_path = ExportUtils.save_depth_npy(filepath, depth_np)
        elif format == "exr":
            saved_path = ExportUtils.save_depth_exr(filepath, depth_np)
        elif format == "pfm":
            saved_path = ExportUtils.save_depth_pfm(filepath, depth_np)
        elif format == "png16":
            saved_path = ExportUtils.save_depth_png16(filepath, depth_np)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node 8: SaveCameraParams
# ============================================================================

class SaveCameraParams:
    """
    Export camera parameters for 3D reconstruction tools.
    Supports: JSON, COLMAP, NPY
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_poses": ("POSES", {
                    "tooltip": "Camera pose matrices from the HWM Inference node. Each pose is a 4x4 matrix describing camera position and orientation in 3D space for each frame."
                }),
                "camera_intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsic parameters from the HWM Inference node. 3x3 matrix containing focal length, principal point, and other internal camera properties."
                }),
                "filepath": ("STRING", {
                    "default": "./output/cameras.json",
                    "multiline": False,
                    "tooltip": "Where to save camera parameters. JSON format saves in one readable file. NPY format saves as two files: filepath_poses.npy and filepath_intrinsics.npy."
                }),
                "format": (["json", "npy"], {
                    "default": "json",
                    "tooltip": "File format for camera data. JSON is human-readable and good for debugging. NPY is binary format for fast loading in Python/NumPy scripts."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        filepath: str,
        format: str
    ) -> Tuple[str]:
        """Save camera parameters to file."""

        # Check if camera parameters are available
        if camera_poses is None:
            print("⚠ Warning: Camera poses not available - skipping export")
            return ("",)

        if camera_intrinsics is None:
            print("⚠ Warning: Camera intrinsics not available - skipping export")
            print("  Note: Intrinsics may require specific model configuration")
            return ("",)

        # Convert to numpy
        poses_np = tensor_to_numpy(camera_poses)
        intrinsics_np = tensor_to_numpy(camera_intrinsics)

        # Save based on format
        if format == "json":
            if not filepath.endswith('.json'):
                filepath = filepath.rsplit('.', 1)[0] + '.json'
            saved_path = ExportUtils.save_camera_parameters_json(
                filepath, poses_np, intrinsics_np
            )
        elif format == "npy":
            # Remove extension for NPY (will add _poses.npy and _intrinsics.npy)
            base_path = filepath.rsplit('.', 1)[0]
            saved_path = ExportUtils.save_camera_parameters_npy(
                base_path, poses_np, intrinsics_np
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node 9: SaveCOLMAPReconstruction
# ============================================================================

class SaveCOLMAPReconstruction:
    """
    Export COLMAP reconstruction for Structure-from-Motion pipelines.
    Creates camera poses, intrinsics, and 3D point cloud in COLMAP format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pts3d": ("PTS3D", {
                    "tooltip": "3D points from HWM Inference. Dense point cloud will be converted to sparse COLMAP format."
                }),
                "camera_poses": ("POSES", {
                    "tooltip": "Camera pose matrices (4x4) from HWM Inference. Describes camera position and orientation for each frame."
                }),
                "camera_intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsic matrices (3x3) from HWM Inference. Contains focal length and principal point."
                }),
                "output_dir": ("STRING", {
                    "default": "./output/colmap",
                    "multiline": False,
                    "tooltip": "Directory to save COLMAP reconstruction. Will create cameras.bin, images.bin, and points3D.bin files."
                }),
                "camera_model": (["SIMPLE_PINHOLE", "PINHOLE"], {
                    "default": "SIMPLE_PINHOLE",
                    "tooltip": "COLMAP camera model. SIMPLE_PINHOLE: single focal length. PINHOLE: separate fx/fy focal lengths."
                }),
                "shared_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Share camera parameters across all frames. True=assume same camera for all images. False=allow different cameras per frame."
                }),
                "subsample_factor": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Downsample dense points by this factor. 4=keep every 4th point. Higher values create smaller, faster COLMAP reconstructions."
                }),
            },
            "optional": {
                "pts3d_rgb": ("*", {
                    "tooltip": "Optional: RGB colors for 3D points from HWM Inference. If not provided, points will be white."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save_colmap"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save_colmap(
        self,
        pts3d: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        output_dir: str,
        camera_model: str,
        shared_camera: bool,
        subsample_factor: int,
        pts3d_rgb: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """Export COLMAP reconstruction."""
        import os
        from src.utils.build_pycolmap_recon import build_pycolmap_reconstruction

        # Check if required data is available
        if pts3d is None:
            print("⚠ Warning: 3D points not available - skipping COLMAP export")
            return ("",)

        if camera_poses is None:
            print("⚠ Warning: Camera poses not available - skipping COLMAP export")
            return ("",)

        if camera_intrinsics is None:
            print("⚠ Warning: Camera intrinsics not available - skipping COLMAP export")
            print("  Note: COLMAP export requires intrinsics for camera calibration")
            return ("",)

        # Convert to numpy
        pts3d_np = tensor_to_numpy(pts3d)  # (B, H, W, 3)
        poses_np = tensor_to_numpy(camera_poses)  # (B, 4, 4)
        intrinsics_np = tensor_to_numpy(camera_intrinsics)  # (B, 3, 3)

        B, H, W, _ = pts3d_np.shape

        # Flatten dense points to sparse list
        pts3d_flat = pts3d_np.reshape(-1, 3)  # (B*H*W, 3)

        # Subsample points to reduce size
        subsample_mask = np.arange(len(pts3d_flat)) % subsample_factor == 0
        pts3d_sparse = pts3d_flat[subsample_mask]

        # Generate pixel coordinates (x, y, frame_idx)
        pixel_coords = []
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    idx = b * H * W + h * W + w
                    if subsample_mask[idx]:
                        pixel_coords.append([w, h, b])  # x, y, frame_idx
        pixel_coords = np.array(pixel_coords, dtype=np.float32)

        # Handle colors
        if pts3d_rgb is not None:
            rgb_np = tensor_to_numpy(pts3d_rgb)  # (B, H, W, 3)
            rgb_flat = rgb_np.reshape(-1, 3)
            rgb_sparse = (rgb_flat[subsample_mask] * 255).astype(np.uint8)
        else:
            # Default to white
            rgb_sparse = np.full((len(pts3d_sparse), 3), 255, dtype=np.uint8)

        # Filter out invalid points (NaN, Inf)
        valid_mask = np.isfinite(pts3d_sparse).all(axis=1)
        pts3d_sparse = pts3d_sparse[valid_mask]
        pixel_coords = pixel_coords[valid_mask]
        rgb_sparse = rgb_sparse[valid_mask]

        print(f"COLMAP Export: {len(pts3d_sparse)} points, {B} frames")

        # Build COLMAP reconstruction
        try:
            reconstruction = build_pycolmap_reconstruction(
                points=pts3d_sparse,
                pixel_coords=pixel_coords,
                point_colors=rgb_sparse,
                poses=poses_np,
                intrinsics=intrinsics_np,
                image_size=(W, H),
                shared_camera_model=shared_camera,
                camera_model=camera_model
            )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Write COLMAP binary format
            reconstruction.write(output_dir)

            print(f"Saved COLMAP reconstruction to: {output_dir}")
            return (output_dir,)

        except Exception as e:
            print(f"Error creating COLMAP reconstruction: {e}")
            import traceback
            traceback.print_exc()
            raise


# ============================================================================
# Node 10: View3DInBrowser
# ============================================================================

class View3DInBrowser:
    """
    Launch an interactive 3D viewer in the browser for point clouds and Gaussian splats.

    Opens a WebGL-based viewer using Three.js to visualize .ply and .splat files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the .ply or .splat file to visualize. Can be a point cloud or Gaussian splat file from SavePointCloud or Save3DGaussians nodes."
                }),
                "mode": (["auto", "pointcloud", "splat"], {
                    "default": "auto",
                    "tooltip": "Rendering mode. 'auto' detects from file content, 'pointcloud' renders as points, 'splat' renders as Gaussian splats with transparency."
                }),
                "auto_open": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically open the viewer in your default web browser when the node executes."
                }),
                "port": ("INT", {
                    "default": 8765,
                    "min": 1024,
                    "max": 65535,
                    "tooltip": "Port number for the local web server. Default is 8765. Change if port is already in use."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("viewer_url",)
    FUNCTION = "view"
    CATEGORY = "HunyuanWorld-Mirror/visualization"
    OUTPUT_NODE = True

    def view(
        self,
        file_path: str,
        mode: str,
        auto_open: bool,
        port: int
    ) -> Tuple[str]:
        """Launch the 3D viewer in browser."""
        from .viewer_server import open_viewer
        import os

        # Validate file path
        if not file_path or not os.path.isfile(file_path):
            print(f"⚠ Warning: File not found or invalid path: {file_path}")
            print("  Skipping viewer launch.")
            return ("",)

        # Auto-detect mode from file content if set to auto
        if mode == "auto":
            if file_path.lower().endswith('.splat'):
                mode = "splat"
            elif file_path.lower().endswith('.ply'):
                # Check PLY header for Gaussian attributes
                try:
                    with open(file_path, 'rb') as f:
                        # Read PLY header (ASCII until 'end_header')
                        header = []
                        for _ in range(100):  # Limit header reading
                            line = f.readline().decode('ascii', errors='ignore').strip()
                            header.append(line)
                            if line == 'end_header':
                                break

                        header_text = ' '.join(header)
                        # Check for Gaussian-specific attributes
                        has_scale = 'scale_0' in header_text
                        has_rotation = 'rot_0' in header_text or 'rotation' in header_text
                        has_opacity = 'opacity' in header_text

                        if has_scale and has_rotation:
                            mode = "splat"
                            print(f"  Detected Gaussian splat attributes in PLY")
                        else:
                            mode = "pointcloud"
                except Exception as e:
                    print(f"  Warning: Could not read PLY header, defaulting to pointcloud mode: {e}")
                    mode = "pointcloud"
            else:
                # Default to pointcloud for other files
                mode = "pointcloud"

        print("\n" + "=" * 70)
        print("Launching 3D Viewer")
        print("=" * 70)

        try:
            # Open viewer
            url = open_viewer(
                file_path=file_path,
                mode=mode,
                port=port,
                auto_open=auto_open
            )

            print(f"✓ 3D Viewer ready")
            print(f"  File: {os.path.basename(file_path)}")
            print(f"  Mode: {mode}")
            print(f"  URL: {url}")

            if not auto_open:
                print(f"\n  ℹ Auto-open disabled. Open this URL in your browser:")
                print(f"    {url}\n")

            print("=" * 70 + "\n")

            return (url,)

        except Exception as e:
            print(f"✗ Error launching viewer: {e}")
            import traceback
            traceback.print_exc()
            return ("",)


# ============================================================================
# Node: SkySegmentation
# ============================================================================

class SkySegmentation:
    """
    Segments sky from images using Tencent's official demo's ONNX model
    (JianyuanWang/skyseg, via visual_util.py::segment_sky -- ported directly).

    Matters for outdoor scenes: unmasked sky pixels have no real depth (the
    model still has to guess *something*), producing unconstrained/runaway
    geometry. Feed this mask into HWMSavePointCloud/SaveCompositeScene's
    sky_mask input to filter it out before export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source images to segment (the same batch fed into HWMInference)."
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("sky_mask",)
    FUNCTION = "segment"
    CATEGORY = "HunyuanWorld-Mirror/preprocessing"

    _onnx_session = None

    @classmethod
    def _get_session(cls):
        if cls._onnx_session is not None:
            return cls._onnx_session
        import onnxruntime
        import folder_paths
        model_path = os.path.join(folder_paths.models_dir, "skyseg", "skyseg.onnx")
        download_skyseg_model(model_path)
        cls._onnx_session = onnxruntime.InferenceSession(model_path)
        return cls._onnx_session

    def segment(self, images: torch.Tensor) -> Tuple[torch.Tensor]:
        session = self._get_session()
        images_np = tensor_to_numpy(images)  # [N, H, W, 3] float [0,1]

        masks = []
        for i in range(images_np.shape[0]):
            keep_mask = segment_sky_mask(images_np[i], session)  # HxW bool, True=non-sky
            masks.append(torch.from_numpy(keep_mask.astype(np.float32)))

        return (torch.stack(masks, dim=0),)


# ============================================================================
# Node: RenderGaussianFlythrough
# ============================================================================

class RenderGaussianFlythrough:
    """
    Renders an actual camera-flythrough video through the reconstructed
    Gaussians, using the model's own real-time rasterizer (model.gs_renderer),
    ported from Tencent's official demo (src/utils/render_utils.py --
    render_interpolated_video).

    This is the real gap between "export a raw Gaussian PLY and hope a
    third-party WebGL viewer interprets it correctly" and how the official
    demo actually presents its results: a properly rendered video, not a
    static point dump handed to an external renderer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HWMIRROR_MODEL", {
                    "tooltip": "The loaded model from LoadHunyuanWorldMirrorModel -- its gs_renderer does the actual rendering."
                }),
                "gaussians": ("GAUSSIANS", {
                    "tooltip": "Gaussian splat parameters from HWMInference."
                }),
                "camera_poses": ("POSES", {
                    "tooltip": "Camera-to-world matrices from HWMInference."
                }),
                "camera_intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsic matrices from HWMInference."
                }),
                "width": ("INT", {"default": 518, "min": 64, "max": 4096, "tooltip": "Render width -- should match the preprocessed image width."}),
                "height": ("INT", {"default": 518, "min": 64, "max": 4096, "tooltip": "Render height -- should match the preprocessed image height."}),
                "interp_per_pair": ("INT", {
                    "default": 20, "min": 1, "max": 120,
                    "tooltip": "Interpolated frames inserted between each pair of real camera poses (multi-view input), or density of the synthesized orbit for single-image input."
                }),
                "loop_reverse": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append the reversed sequence so the video loops seamlessly back to its start."
                }),
                "apply_spread_effect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Play the 'Spread' reveal effect (the scene materializes outward from center with a color trail) continuously across the flythrough -- the one cosmetic flourish actually wired up in the official demo's own gs_effects.py."
                }),
                "effect_speed": ("FLOAT", {
                    "default": 0.04, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "How fast the Spread effect's internal clock advances per frame. Official demo default: 0.04."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb_frames", "depth_frames")
    FUNCTION = "render"
    CATEGORY = "HunyuanWorld-Mirror/output"

    def render(
        self,
        model,
        gaussians: Dict[str, torch.Tensor],
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        width: int,
        height: int,
        interp_per_pair: int,
        loop_reverse: bool,
        apply_spread_effect: bool,
        effect_speed: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gaussians is None or gaussians.get('means') is None:
            print("Warning: Gaussian parameters not available - skipping flythrough render")
            placeholder = torch.zeros((1, height, width, 3))
            return (placeholder, placeholder)

        gs_renderer = getattr(model, "gs_renderer", None)
        if gs_renderer is None:
            raise ValueError(
                "model has no gs_renderer attribute -- RenderGaussianFlythrough needs the "
                "raw WorldMirror model instance from LoadHunyuanWorldMirrorModel, not a "
                "pre-processed gaussians dict."
            )

        # Same colors-from-SH-DC-term fallback as Save3DGaussians (nodes.py's
        # own save() method) -- HWMInference's raw gaussians dict has
        # colors=None; the real per-Gaussian RGB lives in the first 3 channels
        # of 'sh' (already plain [0,1] RGB here, not a real spherical-harmonics
        # basis needing the SH_C0 decode -- confirmed by that existing,
        # previously-verified code path, so mirrored verbatim rather than
        # re-derived).
        colors = gaussians.get('colors', None)
        if colors is None:
            sh_tensor = gaussians.get('sh', None)
            if sh_tensor is not None and sh_tensor.shape[-1] >= 3:
                colors = sh_tensor[..., :3]
        if colors is None:
            raise ValueError("gaussians dict has neither 'colors' nor a usable 'sh' DC term -- cannot render.")
        # 'sh' carries a singleton SH-coefficient axis (degree 0 -> shape
        # [..., N, 1, 3]) that numpy code elsewhere flattens away harmlessly
        # via .reshape(-1, 3) at export time -- squeeze it explicitly here
        # since nothing downstream in this torch path does that reshape.
        if colors.dim() == 4 and colors.shape[-2] == 1:
            colors = colors.squeeze(-2)

        # Ensure batch dimension: gaussians as produced by HWMInference already
        # carry [1, N, ...]; camera_poses/intrinsics too.
        means = gaussians['means']
        if means.dim() == 2:
            means = means.unsqueeze(0)
        # 'opacities' is scalar-per-point (no channel axis), so a batched
        # tensor is 2D ([1, N]) where means/quats/scales/colors are 3D when
        # batched ([1, N, C]) -- unbatched is one dim lower in each case.
        # Blanket-checking dim()==2 for all of them double-batches opacities.
        splats = {
            'means': means,
            'quats': gaussians['quats'].unsqueeze(0) if gaussians['quats'].dim() == 2 else gaussians['quats'],
            'scales': gaussians['scales'].unsqueeze(0) if gaussians['scales'].dim() == 2 else gaussians['scales'],
            'opacities': gaussians['opacities'].unsqueeze(0) if gaussians['opacities'].dim() == 1 else gaussians['opacities'],
            'colors': colors.unsqueeze(0) if colors.dim() == 2 else colors,
        }

        cam_poses = camera_poses if camera_poses.dim() == 4 else camera_poses.unsqueeze(0)
        cam_intr = camera_intrinsics if camera_intrinsics.dim() == 4 else camera_intrinsics.unsqueeze(0)

        rgb_frames, depth_frames = render_gaussian_flythrough(
            gs_renderer=gs_renderer,
            splats=splats,
            camtoworlds=cam_poses,
            intrinsics=cam_intr,
            hw=(height, width),
            interp_per_pair=interp_per_pair,
            loop_reverse=loop_reverse,
            apply_spread_effect=apply_spread_effect,
            effect_speed=effect_speed,
        )

        return (rgb_frames.cpu().float(), depth_frames.cpu().float())


# ============================================================================
# Node: SaveCompositeScene
# ============================================================================

class SaveCompositeScene:
    """
    Assembles a single GLB scene from the raw per-pixel reconstruction (a
    real textured mesh per view, or a merged point cloud) plus color-coded
    camera-frustum markers for every view -- ported from Tencent's official
    demo (src/utils/visual_util.py::convert_predictions_to_glb_scene).

    This is the "stage it, animate a virtual camera" deliverable: a GLB you
    can actually drop into Blender, the Three.js Editor, or 3DGenStudio's own
    mesh pipeline, with real camera objects included -- not a Gaussian splat
    PLY that needs special-case viewer support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points3d": ("POINTS3D", {
                    "tooltip": "3D point coordinates from HWMInference."
                }),
                "images": ("IMAGE", {
                    "tooltip": "Source images (for mesh/point-cloud vertex colors)."
                }),
                "camera_poses": ("POSES", {
                    "tooltip": "Camera-to-world matrices from HWMInference -- placed into the scene as frustum markers."
                }),
                "filepath": ("STRING", {
                    "default": "./output/hwm_composite_scene.glb",
                    "multiline": False,
                }),
                "as_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Build a real textured mesh per view (True) instead of a single merged point cloud (False)."
                }),
                "show_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add a color-coded frustum marker mesh for each camera view."
                }),
            },
            "optional": {
                "normals": ("NORMALS", {
                    "tooltip": "Optional surface normals, same shape as points3d -- improves mesh shading."
                }),
                "sky_mask": ("MASK", {
                    "tooltip": "Optional sky keep-mask from SkySegmentation -- filters sky pixels out of the mesh/point cloud."
                }),
                "depth": ("DEPTH", {
                    "tooltip": "Optional depth map -- enables apply_edge_mask, same mechanism as HWMSavePointCloud."
                }),
                "apply_edge_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove points at depth/normal discontinuities (requires both depth and normals connected)."
                }),
                "edge_normal_threshold": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 90.0, "step": 0.5}),
                "edge_depth_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.005}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        points3d: torch.Tensor,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        filepath: str,
        as_mesh: bool,
        show_camera: bool,
        normals: Optional[torch.Tensor] = None,
        sky_mask: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        apply_edge_mask: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
    ) -> Tuple[str]:
        if points3d is None:
            print("Warning: 3D points not available - skipping composite scene export")
            return ("",)

        points_np = tensor_to_numpy(points3d)
        if points_np.ndim == 5:
            points_np = points_np.squeeze(0)  # [S, H, W, 3]

        images_np = tensor_to_numpy(images)  # [S, H, W, 3] float [0,1]
        poses_np = tensor_to_numpy(camera_poses)
        if poses_np.ndim == 4 and poses_np.shape[0] == 1:
            poses_np = poses_np[0]  # [S, 4, 4]

        s, h, w = points_np.shape[:3]

        normals_np = None
        if normals is not None:
            normals_np = tensor_to_numpy(normals)
            if normals_np.ndim == 5:
                normals_np = normals_np.squeeze(0)

        valid_mask = np.ones((s, h, w), dtype=bool)

        if sky_mask is not None:
            sky_np = tensor_to_numpy(sky_mask).astype(bool)
            if sky_np.shape[-2:] == (h, w):
                valid_mask &= sky_np

        if apply_edge_mask and depth is not None and normals_np is not None:
            depth_np = tensor_to_numpy(depth)
            if depth_np.ndim == 5:
                depth_np = depth_np.squeeze(0)
                if depth_np.shape[-1] == 1:
                    depth_np = depth_np.squeeze(-1)
            elif depth_np.ndim == 4 and depth_np.shape[-1] == 1:
                depth_np = depth_np.squeeze(-1)

            if depth_np.shape[0] == s and depth_np.shape[-2:] == (h, w):
                for i in range(s):
                    d_edges = depth_edge(depth_np[i], rtol=edge_depth_threshold)
                    n_edges = normals_edge(normals_np[i], tol=edge_normal_threshold)
                    valid_mask[i] &= ~(d_edges & n_edges)
            else:
                print(f"  Warning: depth shape {depth_np.shape} doesn't match points3d grid ({s},{h},{w}) -- skipping edge mask")

        scene = build_composite_scene(
            points3d=points_np,
            images=images_np,
            camera_poses=poses_np,
            normals=normals_np,
            valid_masks=valid_mask,
            show_camera=show_camera,
            as_mesh=as_mesh,
        )

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if not filepath.endswith(".glb"):
            filepath = filepath.rsplit(".", 1)[0] + ".glb"
        scene.export(filepath, file_type="glb")

        return (filepath,)


# ============================================================================
# Node Mappings for ComfyUI Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PreprocessImagesForHWM": PreprocessImagesForHWM,
    "LoadHunyuanWorldMirrorModel": LoadHunyuanWorldMirrorModel,
    "HWMInference": HWMInference,
    "VisualizeDepth": VisualizeDepth,
    "VisualizeNormals": VisualizeNormals,
    # Registered as "HWMSavePointCloud", not "SavePointCloud" -- ComfyUI core added
    # its own native SavePointCloud node (comfy_extras/nodes_save_3d.py) after this
    # pack was written, an exact NODE_CLASS_MAPPINGS key collision, and core's wins
    # at registration time. Renaming the key (not the class name) is the minimal fix.
    "HWMSavePointCloud": SavePointCloud,
    "Save3DGaussians": Save3DGaussians,
    "SaveDepthMap": SaveDepthMap,
    "SaveCameraParams": SaveCameraParams,
    "SaveCOLMAPReconstruction": SaveCOLMAPReconstruction,
    "View3DInBrowser": View3DInBrowser,
    "SkySegmentation": SkySegmentation,
    "RenderGaussianFlythrough": RenderGaussianFlythrough,
    "SaveCompositeScene": SaveCompositeScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreprocessImagesForHWM": "Preprocess Images for HWM",
    "LoadHunyuanWorldMirrorModel": "Load HunyuanWorld-Mirror Model",
    "HWMInference": "HWM Inference",
    "VisualizeDepth": "Visualize Depth",
    "VisualizeNormals": "Visualize Normals",
    "HWMSavePointCloud": "Save Point Cloud (HWM)",
    "Save3DGaussians": "Save 3D Gaussians",
    "SaveDepthMap": "Save Depth Map",
    "SaveCameraParams": "Save Camera Parameters",
    "SaveCOLMAPReconstruction": "Save COLMAP Reconstruction",
    "View3DInBrowser": "View 3D in Browser",
    "SkySegmentation": "Sky Segmentation",
    "RenderGaussianFlythrough": "Render Gaussian Flythrough",
    "SaveCompositeScene": "Save Composite Scene (GLB)",
}

# Web directory for custom frontend extensions (if needed in the future)
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")
