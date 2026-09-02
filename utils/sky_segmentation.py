"""
Sky segmentation, ported from Tencent's official HunyuanWorld-Mirror demo
(huggingface.co/spaces/tencent/HunyuanWorld-Mirror, src/utils/visual_util.py --
segment_sky/run_skyseg -- fetched and ported directly, not reimplemented from
scratch, learning from the quaternion-convention bug elsewhere in this pack).

Matters for outdoor scenes: unmasked sky pixels have no real depth (the model
still has to guess *something*), which is unconstrained/runaway geometry the
official demo explicitly filters out before export via mask_sky_bg.
"""

import copy
import os

import cv2
import numpy as np
import requests

SKYSEG_MODEL_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


def download_skyseg_model(destination_path: str) -> str:
    """Downloads skyseg.onnx if not already present at destination_path."""
    if os.path.isfile(destination_path):
        return destination_path

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    response = requests.get(SKYSEG_MODEL_URL, allow_redirects=True, stream=True, timeout=60)
    response.raise_for_status()
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return destination_path


def run_skyseg(onnx_session, input_size, image_bgr: np.ndarray) -> np.ndarray:
    """
    Runs sky segmentation inference using the ONNX model.

    Args:
        onnx_session: onnxruntime.InferenceSession with the skyseg model loaded
        input_size: [width, height] the model expects (320x320 for this model)
        image_bgr: input image in BGR format (as returned by cv2.imread)

    Returns:
        np.ndarray: raw segmentation map, uint8, low values = sky
    """
    temp_image = copy.deepcopy(image_bgr)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def segment_sky_mask(image_rgb_float: np.ndarray, onnx_session) -> np.ndarray:
    """
    Args:
        image_rgb_float: HxWx3 RGB image, float [0, 1] (ComfyUI's own IMAGE convention)
        onnx_session: onnxruntime.InferenceSession with the skyseg model loaded

    Returns:
        np.ndarray: HxW bool mask, True = non-sky (keep), False = sky (discard) --
        matches the official demo's convention (sky_mask used as a keep-mask via
        `valid_points_mask & sky_region_mask`).
    """
    image_bgr = cv2.cvtColor((image_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    result_map = run_skyseg(onnx_session, [320, 320], image_bgr)
    result_map_original = cv2.resize(result_map, (image_bgr.shape[1], image_bgr.shape[0]))

    # Model outputs LOW values for sky, high values for non-sky.
    return result_map_original >= 32
