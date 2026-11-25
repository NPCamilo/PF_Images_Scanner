"""Utilidades matriciales de Imagen 2D <-> Señal 1D / Bitstream y filtro rojo dominante."""

from typing import Literal, Tuple
import numpy as np

from dsp_engine.contracts import ImageMatrix


def image_to_signal_vector(image: ImageMatrix) -> np.ndarray:
    """Convierte ImageMatrix en señal 1D normalizada [-1, 1] mediante luminancia ITU-R BT.601."""
    total = image.width * image.height
    if image.channels == 1:
        lum = image.pixels[:total].astype(np.float64)
    else:
        reshaped = image.pixels[: total * image.channels].reshape(total, image.channels).astype(np.float64)
        lum = 0.299 * reshaped[:, 0] + 0.587 * reshaped[:, 1] + 0.114 * reshaped[:, 2]
    return ((lum / 127.5) - 1.0).astype(np.float64)


def image_to_bitstream(image: ImageMatrix, bit_depth: int = 8) -> np.ndarray:
    """Serializa los canales espaciales (1 o RGB) de ImageMatrix a un flujo plano de bits."""
    total = image.width * image.height
    out_channels = 1 if image.channels == 1 else 3
    max_val = (1 << bit_depth) - 1

    if image.channels == 1:
        vals = image.pixels[:total].astype(np.float64)
    else:
        vals = (
            image.pixels[: total * image.channels]
            .reshape(total, image.channels)[:, :3]
            .reshape(-1)
            .astype(np.float64)
        )

    scaled = np.round((vals / 255.0) * max_val).astype(np.int32)
    shifts = np.arange(bit_depth - 1, -1, -1, dtype=np.int32)
    bits = ((scaled[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return bits


def bitstream_to_image(
    bits: np.ndarray,
    width: int,
    height: int,
    source_channels: Literal[1, 3] = 3,
    bit_depth: int = 8,
) -> ImageMatrix:
    """Reconstruye una ImageMatrix RGBA a partir de un flujo plano de bits."""
    total_pixels = width * height
    expected_values = total_pixels * source_channels
    expected_bits = expected_values * bit_depth
    max_val = max(1, (1 << bit_depth) - 1)

    flat_bits = np.zeros(expected_bits, dtype=np.uint8)
    copy_len = min(len(bits), expected_bits)
    flat_bits[:copy_len] = np.asarray(bits[:copy_len], dtype=np.uint8) & 1

    weights = 1 << np.arange(bit_depth - 1, -1, -1, dtype=np.int32)
    quant_vals = flat_bits.reshape(expected_values, bit_depth) @ weights
    rgb_vals = np.clip(np.round((quant_vals / max_val) * 255.0), 0, 255).astype(np.uint8)

    rgba = np.zeros((total_pixels, 4), dtype=np.uint8)
    if source_channels == 1:
        rgba[:, 0] = rgb_vals
        rgba[:, 1] = rgb_vals
        rgba[:, 2] = rgb_vals
    else:
        rgba[:, :3] = rgb_vals.reshape(total_pixels, 3)
    rgba[:, 3] = 255

    return ImageMatrix(
        width=width,
        height=height,
        channels=4,
        pixels=rgba.reshape(-1),
        label=f"Reconstructed-{width}x{height}",
    )


def extract_red_dominant_mask(image: ImageMatrix) -> Tuple[ImageMatrix, float]:
    """Aísla píxeles con dominancia de canal rojo y atenúa el fondo en escala de grises."""
    total = image.width * image.height
    if image.channels == 1:
        r = g = b = image.pixels[:total].astype(np.float64)
    else:
        arr = image.pixels[: total * image.channels].reshape(total, image.channels).astype(np.float64)
        r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]

    is_red = (r > 120.0) & (r > g * 1.45) & (r > b * 1.45)
    gray_dim = np.round((0.299 * r + 0.587 * g + 0.114 * b) * 0.35)

    out = np.zeros((total, 4), dtype=np.uint8)
    out[:, 0] = np.where(is_red, r, gray_dim).astype(np.uint8)
    out[:, 1] = np.where(is_red, np.round(g * 0.25), gray_dim).astype(np.uint8)
    out[:, 2] = np.where(is_red, np.round(b * 0.25), gray_dim).astype(np.uint8)
    out[:, 3] = 255

    red_ratio = float(np.sum(is_red) / total) if total > 0 else 0.0
    return (
        ImageMatrix(
            width=image.width,
            height=image.height,
            channels=4,
            pixels=out.reshape(-1),
            label=f"{image.label}-RedMask",
        ),
        red_ratio,
    )
