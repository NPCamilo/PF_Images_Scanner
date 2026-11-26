"""
Compresión de Imagen con Pérdida mediante DCT 2D por Bloques 8x8 y Cuantización JPEG.
Responsable: Mariano Orrego (feat/compression-docs-mariano)

Ecuaciones Matemáticas (DCT-II Ortonormal 8x8):
    C_{k, n} = alpha(k) * cos( pi * (2n + 1) * k / 16 )
    donde alpha(0) = 1/sqrt(8) y alpha(k) = 1/2 para k = 1..7.
- Transformada Directa 2D: Y = C * (X - 128) * C^T
- Cuantización: Y_q[u, v] = round( Y[u, v] / Q[u, v] )
- Transformada Inversa 2D: X_rec = clip( C^T * (Y_q * Q) * C + 128, 0, 255 )
- Serialización: Escaneo Zig-Zag + RLE de ceros con marcador EOB (End-Of-Block).
"""

import struct
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import CompressedPayload, ImageMatrix

# Matriz estándar JPEG Luminancia (Q50)
JPEG_LUMINANCE_Q50: NDArray[np.float64] = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)

# Matriz estándar JPEG Crominancia (Q50)
JPEG_CHROMINANCE_Q50: NDArray[np.float64] = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float64,
)

# Índices lineales del recorrido Zig-Zag para bloque 8x8
_ZIGZAG_INDICES: NDArray[np.int32] = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ],
    dtype=np.int32,
)


def _build_dct_matrix_8() -> NDArray[np.float64]:
    """Construye la matriz ortonormal 8x8 de la DCT-II (C @ C.T = I_8)."""
    k = np.arange(8, dtype=np.float64)[:, None]
    n = np.arange(8, dtype=np.float64)[None, :]
    alpha = np.where(k == 0, 1.0 / np.sqrt(8.0), 0.5)
    return alpha * np.cos((np.pi * (2.0 * n + 1.0) * k) / 16.0)


_DCT_C8: NDArray[np.float64] = _build_dct_matrix_8()


def scale_quantization_matrix(
    base_matrix: NDArray[np.float64],
    quality: int = 75,
    quant_matrix_scale: Optional[float] = None,
) -> NDArray[np.float64]:
    """Escala una matriz de cuantización 8x8 según calidad JPEG [1..100] o factor directo."""
    if quant_matrix_scale is not None:
        if quant_matrix_scale <= 0.0:
            raise ValueError("quant_matrix_scale debe ser estrictamente positivo.")
        return np.clip(np.round(base_matrix * quant_matrix_scale), 1.0, 255.0).astype(np.float64)

    q = max(1, min(100, int(quality)))
    factor = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    scaled = np.floor((base_matrix * factor + 50.0) / 100.0)
    return np.clip(scaled, 1.0, 255.0).astype(np.float64)


def dct2d_blocks_forward(blocks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula la DCT 2D ortonormal sobre un lote de bloques (N, 8, 8)."""
    return _DCT_C8 @ blocks @ _DCT_C8.T


def dct2d_blocks_inverse(coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula la IDCT 2D ortonormal sobre un lote de bloques (N, 8, 8)."""
    return _DCT_C8.T @ coeffs @ _DCT_C8


def _pad_to_8(plane: NDArray[np.float64]) -> Tuple[NDArray[np.float64], int, int]:
    """Rellena por replicación de bordes hasta múltiplos de 8."""
    h, w = plane.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return plane, h, w
    padded = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, h, w


def _plane_to_blocks(plane: NDArray[np.float64]) -> Tuple[NDArray[np.float64], int, int]:
    """Divide un plano 2D múltiplo de 8 en bloques (N, 8, 8)."""
    h, w = plane.shape
    bh, bw = h // 8, w // 8
    blocks = plane.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return blocks, bh, bw


def _blocks_to_plane(blocks: NDArray[np.float64], bh: int, bw: int) -> NDArray[np.float64]:
    """Reconstruye un plano 2D desde bloques (N, 8, 8)."""
    return blocks.reshape(bh, bw, 8, 8).transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)


def _encode_quantized_blocks_rle(q_blocks: NDArray[np.int16]) -> bytes:
    """Serializa bloques 8x8 cuantizados en orden Zig-Zag + Zero-RLE con marcador EOB."""
    buf = bytearray()
    n_blocks = q_blocks.shape[0]
    flat_blocks = q_blocks.reshape(n_blocks, 64)

    for b_idx in range(n_blocks):
        zz = flat_blocks[b_idx][_ZIGZAG_INDICES]
        nz_positions = np.flatnonzero(zz)
        if nz_positions.size == 0:
            buf.append(0x00)  # EOB inmediato
            continue

        last_nz = int(nz_positions[-1])
        zeros = 0
        for pos in range(last_nz + 1):
            val = int(zz[pos])
            if val == 0:
                zeros += 1
                if zeros == 16:
                    buf.append(0xF0)  # ZRL: 16 ceros consecutivos
                    zeros = 0
            else:
                val = max(-32768, min(32767, val))
                if -128 <= val <= 127:
                    buf.append((zeros << 4) | 0x01)
                    buf.extend(struct.pack(">b", val))
                else:
                    buf.append((zeros << 4) | 0x02)
                    buf.extend(struct.pack(">h", val))
                zeros = 0

        if last_nz < 63:
            buf.append(0x00)  # EOB

    return bytes(buf)


def _decode_quantized_blocks_rle(raw: bytes, n_blocks: int) -> NDArray[np.int16]:
    """Decodifica el flujo Zig-Zag + Zero-RLE en bloques (n_blocks, 8, 8) int16."""
    out = np.zeros((n_blocks, 64), dtype=np.int16)
    mv = memoryview(raw)
    idx = 0
    total_len = len(mv)

    for b_idx in range(n_blocks):
        pos = 0
        while idx < total_len and pos < 64:
            tag = mv[idx]
            idx += 1
            if tag == 0x00:
                break  # EOB
            if tag == 0xF0:
                pos += 16
                continue
            zeros = (tag >> 4) & 0x0F
            size_code = tag & 0x0F
            pos += zeros
            if pos >= 64:
                break
            if size_code == 1 and idx < total_len:
                val = struct.unpack_from(">b", mv, idx)[0]
                idx += 1
            elif size_code == 2 and idx + 1 < total_len:
                val = struct.unpack_from(">h", mv, idx)[0]
                idx += 2
            else:
                raise ValueError("Flujo DCT Zig-Zag RLE corrupto o truncado.")
            out[b_idx, _ZIGZAG_INDICES[pos]] = val
            pos += 1

    return out.reshape(n_blocks, 8, 8)


def _rgb_to_ycbcr(rgb: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convierte arreglo (H, W, 3) RGB [0..255] a planos (Y, Cb, Cr) ITU-R BT.601."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def _ycbcr_to_rgb(y: NDArray[np.float64], cb: NDArray[np.float64], cr: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Convierte planos (Y, Cb, Cr) de vuelta a matriz (H, W, 3) uint8 [0..255]."""
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    stacked = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(stacked), 0.0, 255.0).astype(np.uint8)


def compress_dct_image(
    image: ImageMatrix,
    quality: int = 75,
    quant_matrix: Optional[NDArray[np.float64]] = None,
    quant_matrix_scale: Optional[float] = None,
) -> CompressedPayload:
    """
    Comprime una ImageMatrix mediante DCT 2D por bloques 8x8 y matriz de cuantización configurable.
    Retorna CompressedPayload con codec='dct-8x8' y métricas CR / ahorro porcentual.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Las dimensiones de la imagen deben ser positivas.")
    if image.channels not in (1, 3, 4):
        raise ValueError(f"Canales no soportados para DCT: {image.channels}")

    if quant_matrix is not None:
        q_lum = np.asarray(quant_matrix, dtype=np.float64)
        if q_lum.shape != (8, 8) or np.any(q_lum <= 0):
            raise ValueError("La matriz de cuantización personalizada debe tener forma (8, 8) y valores > 0.")
        q_chr = q_lum.copy()
    else:
        q_lum = scale_quantization_matrix(JPEG_LUMINANCE_Q50, quality=quality, quant_matrix_scale=quant_matrix_scale)
        q_chr = scale_quantization_matrix(JPEG_CHROMINANCE_Q50, quality=quality, quant_matrix_scale=quant_matrix_scale)

    total_px = image.width * image.height
    expected_len = total_px * image.channels
    if image.pixels.size < expected_len:
        raise ValueError(
            f"Buffer de píxeles insuficiente ({image.pixels.size}) para {image.width}x{image.height}x{image.channels}."
        )
    raw_pixels = np.asarray(image.pixels[:expected_len], dtype=np.uint8)

    if image.channels == 1:
        planes = [raw_pixels.reshape(image.height, image.width).astype(np.float64)]
        q_mats = [q_lum]
        # Tamaño original espacial (1 canal)
        original_bytes = total_px
    else:
        rgb = raw_pixels.reshape(image.height, image.width, image.channels)[:, :, :3].astype(np.float64)
        y, cb, cr = _rgb_to_ycbcr(rgb)
        planes = [y, cb, cr]
        q_mats = [q_lum, q_chr, q_chr]
        original_bytes = total_px * 3

    stream_parts: List[bytes] = []
    plane_byte_lengths: List[int] = []
    padded_bh = 0
    padded_bw = 0

    for plane, q_mat in zip(planes, q_mats):
        padded, _, _ = _pad_to_8(plane)
        blocks, bh, bw = _plane_to_blocks(padded)
        padded_bh, padded_bw = bh, bw
        coeffs = dct2d_blocks_forward(blocks - 128.0)
        q_coeffs = np.round(coeffs / q_mat[None, :, :]).astype(np.int16)
        encoded_plane = _encode_quantized_blocks_rle(q_coeffs)
        stream_parts.append(encoded_plane)
        plane_byte_lengths.append(len(encoded_plane))

    payload_bytes = b"".join(stream_parts)
    payload_arr = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
    compressed_bytes = int(payload_arr.size)
    ratio = float(original_bytes / compressed_bytes) if compressed_bytes > 0 else float(original_bytes)
    savings = float((1.0 - (compressed_bytes / original_bytes)) * 100.0) if original_bytes > 0 else 0.0

    return CompressedPayload(
        codec="dct-8x8",
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        savings_percent=savings,
        payload=payload_arr,
        metadata={
            "sourceType": "image",
            "width": int(image.width),
            "height": int(image.height),
            "channels": int(image.channels),
            "bitDepth": 8,
            "quality": int(quality),
            "quantMatrixScale": float(quant_matrix_scale) if quant_matrix_scale is not None else 1.0,
            "blocksH": padded_bh,
            "blocksW": padded_bw,
            "planeByteLengths": plane_byte_lengths,
            "qLuminance": q_lum.tolist(),
            "qChrominance": q_chr.tolist(),
            "label": image.label,
        },
    )


def decompress_dct_image(compressed: CompressedPayload) -> ImageMatrix:
    """Reconstruye una ImageMatrix mediante dequantización, IDCT 2D 8x8 y recorte al tamaño original."""
    if compressed.codec != "dct-8x8":
        raise ValueError(f"Códec inválido para decompress_dct_image: '{compressed.codec}'.")

    meta = compressed.metadata
    width = int(meta["width"])
    height = int(meta["height"])
    channels = int(meta["channels"])
    bh = int(meta["blocksH"])
    bw = int(meta["blocksW"])
    n_blocks = bh * bw
    plane_lengths: List[int] = [int(x) for x in meta["planeByteLengths"]]

    q_lum = np.asarray(meta["qLuminance"], dtype=np.float64).reshape(8, 8)
    q_chr = np.asarray(meta["qChrominance"], dtype=np.float64).reshape(8, 8)
    q_mats = [q_lum] if channels == 1 else [q_lum, q_chr, q_chr]

    raw_bytes = bytes(np.asarray(compressed.payload, dtype=np.uint8))
    offset = 0
    reconstructed_planes: List[NDArray[np.float64]] = []

    for plane_len, q_mat in zip(plane_lengths, q_mats):
        chunk = raw_bytes[offset : offset + plane_len]
        offset += plane_len
        q_blocks = _decode_quantized_blocks_rle(chunk, n_blocks=n_blocks)
        dequant = q_blocks.astype(np.float64) * q_mat[None, :, :]
        spatial = dct2d_blocks_inverse(dequant) + 128.0
        plane = _blocks_to_plane(spatial, bh=bh, bw=bw)[:height, :width]
        reconstructed_planes.append(plane)

    if channels == 1:
        pixels = np.clip(np.round(reconstructed_planes[0]), 0.0, 255.0).astype(np.uint8).reshape(-1)
        out_channels = 1
    else:
        rgb = _ycbcr_to_rgb(reconstructed_planes[0], reconstructed_planes[1], reconstructed_planes[2])
        if channels == 4:
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb
            rgba[:, :, 3] = 255
            pixels = rgba.reshape(-1)
            out_channels = 4
        else:
            pixels = rgb.reshape(-1)
            out_channels = 3

    return ImageMatrix(
        width=width,
        height=height,
        channels=out_channels,  # type: ignore[arg-type]
        pixels=pixels,
        label=str(meta.get("label", f"DCT-{width}x{height}")),
    )


def compute_image_mse_psnr(original: ImageMatrix, reconstructed: ImageMatrix) -> Tuple[float, float]:
    """Calcula MSE y PSNR (dB) entre imagen original y reconstruida sobre los canales de color."""
    if original.width != reconstructed.width or original.height != reconstructed.height:
        raise ValueError("Las dimensiones de original y reconstruida deben coincidir.")
    total_px = original.width * original.height
    orig_arr = original.pixels[: total_px * original.channels].reshape(total_px, original.channels)
    rec_arr = reconstructed.pixels[: total_px * reconstructed.channels].reshape(total_px, reconstructed.channels)

    cmp_ch = 1 if (original.channels == 1 or reconstructed.channels == 1) else 3
    diff = orig_arr[:, :cmp_ch].astype(np.float64) - rec_arr[:, :cmp_ch].astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-12 else 99.9
    return mse, psnr
