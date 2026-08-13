"""JPEG compression/decompression for remote inference.

Reduces image payload from ~2.76 MB (3 cameras x 480x640x3 uint8) to ~50-100 KB
by resizing to 224x224 and JPEG-encoding at quality 75 before sending over the wire.
The server auto-detects compressed images (bytes vs ndarray) and decodes them.
"""

import cv2
import numpy as np


def _resize_with_pad_hwc(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize an HWC image to target size without distorting aspect ratio."""
    target_h, target_w = target_size
    cur_h, cur_w = image.shape[:2]

    ratio = max(cur_w / target_w, cur_h / target_h)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = cv2.resize(image, (resized_w, resized_h))
    padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

    pad_h0, _ = divmod(target_h - resized_h, 2)
    pad_w0, _ = divmod(target_w - resized_w, 2)
    pad_h1 = pad_h0 + resized_h
    pad_w1 = pad_w0 + resized_w
    padded[pad_h0:pad_h1, pad_w0:pad_w1] = resized
    return padded


def compress_images(
    images: dict, quality: int = 75, resize: tuple[int, int] = (224, 224)
) -> dict:
    """JPEG-encode each camera image array.

    Args:
        images: Dict of camera_name -> (3, H, W) uint8 ndarray (channels-first RGB).
        quality: JPEG quality (0-100). Default 75 matches dataprocessing/image_ops.py.
        resize: Target (H, W) for resizing before encoding.

    Returns:
        Dict of camera_name -> JPEG bytes.
    """
    result = {}
    for name, img in images.items():
        # (3, H, W) channels-first RGB -> (H, W, 3) channels-last
        img_hwc = np.transpose(img, (1, 2, 0))
        # Match the model's standard preprocessing by preserving aspect ratio
        # and padding to the target resolution instead of stretching.
        img_resized = _resize_with_pad_hwc(img_hwc, resize)
        # RGB -> BGR for cv2
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        # JPEG encode
        _, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result[name] = encoded.tobytes()
    return result


def decompress_images(images: dict) -> dict:
    """Decode any JPEG bytes back to uint8 numpy arrays.

    Arrays are passed through unchanged for backward compatibility with
    non-compressed clients.

    Args:
        images: Dict of camera_name -> JPEG bytes or (3, H, W) uint8 ndarray.

    Returns:
        Dict of camera_name -> (3, H, W) uint8 ndarray (channels-first RGB).
    """
    result = {}
    for name, value in images.items():
        if isinstance(value, (bytes, bytearray)):
            # Decode JPEG bytes
            buf = np.frombuffer(value, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # (H, W, 3) -> (3, H, W) channels-first
            result[name] = np.transpose(img_rgb, (2, 0, 1))
        else:
            result[name] = value
    return result
