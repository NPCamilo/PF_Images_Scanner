"""
Compresión Sin Pérdida Run-Length Encoding (RLE — Variante PackBits).
Responsable: Mariano Orrego (feat/compression-docs-mariano)

Ecuación de Tasa de Compresión y Ahorro:
    CR = N_orig / N_comp
    Ahorro (%) = (1 - N_comp / N_orig) * 100

Formato PackBits sobre flujo de bytes (evita expansión 2x en alta entropía):
- Corrida repetida (2..129 bytes iguales): [128 + (run - 2), valor]
- Bloque literal (1..128 bytes distintos): [len - 1, b_0, b_1, ..., b_{len-1}]
Garantiza reconstrucción exacta sin pérdida: MSE = 0, BER = 0%.
"""

from typing import Any, Dict, List, Literal, Optional
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import CompressedPayload, ImageMatrix


def encode_packbits_bytes(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Codifica un arreglo unidimensional uint8 mediante PackBits RLE."""
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    out: List[int] = []
    i = 0
    while i < n:
        run = 1
        val = int(flat[i])
        while i + run < n and int(flat[i + run]) == val and run < 129:
            run += 1

        if run >= 2:
            out.append(128 + (run - 2))
            out.append(val)
            i += run
        else:
            lit_start = i
            lit_len = 0
            while i < n and lit_len < 128:
                if i + 1 < n and int(flat[i]) == int(flat[i + 1]):
                    break
                i += 1
                lit_len += 1
            out.append(lit_len - 1)
            for k in range(lit_start, lit_start + lit_len):
                out.append(int(flat[k]))

    return np.asarray(out, dtype=np.uint8)


def decode_packbits_bytes(
    encoded: NDArray[np.uint8],
    expected_length: Optional[int] = None,
) -> NDArray[np.uint8]:
    """Decodifica un flujo PackBits RLE reconstruyendo exactamente los bytes originales."""
    flat = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    n_in = int(flat.size)
    if n_in == 0 or (expected_length is not None and expected_length <= 0):
        return np.zeros(max(0, expected_length or 0), dtype=np.uint8)

    out: List[int] = []
    idx = 0
    while idx < n_in:
        if expected_length is not None and len(out) >= expected_length:
            break
        ctrl = int(flat[idx])
        idx += 1
        if ctrl >= 128:
            count = (ctrl - 128) + 2
            val = int(flat[idx]) if idx < n_in else 0
            idx += 1
            if expected_length is not None:
                count = min(count, expected_length - len(out))
            out.extend([val] * count)
        else:
            count = ctrl + 1
            for _ in range(count):
                if idx >= n_in:
                    break
                if expected_length is not None and len(out) >= expected_length:
                    break
                out.append(int(flat[idx]))
                idx += 1

    if expected_length is not None and len(out) < expected_length:
        out.extend([0] * (expected_length - len(out)))

    return np.asarray(out, dtype=np.uint8)


def _compute_metrics(original_bytes: int, compressed_bytes: int) -> tuple[float, float]:
    """Calcula CR y ahorro porcentual evitando división por cero."""
    if original_bytes <= 0:
        return 1.0, 0.0
    if compressed_bytes <= 0:
        return float(original_bytes), 100.0
    ratio = float(original_bytes / compressed_bytes)
    savings = float((1.0 - (compressed_bytes / original_bytes)) * 100.0)
    return ratio, savings


def compress_rle(
    data: NDArray[np.uint8],
    source_type: Literal["audio", "image", "bitstream"] = "bitstream",
    metadata: Optional[Dict[str, Any]] = None,
) -> CompressedPayload:
    """Comprime un flujo de bytes mediante RLE PackBits y retorna CompressedPayload."""
    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    encoded = encode_packbits_bytes(raw)
    orig_len = int(raw.size)
    comp_len = int(encoded.size)
    ratio, savings = _compute_metrics(orig_len, comp_len)

    meta: Dict[str, Any] = dict(metadata) if metadata else {}
    meta["sourceType"] = source_type
    meta["originalByteLength"] = orig_len

    return CompressedPayload(
        codec="rle",
        original_bytes=orig_len,
        compressed_bytes=comp_len,
        compression_ratio=ratio,
        savings_percent=savings,
        payload=encoded,
        metadata=meta,
    )


def decompress_rle(compressed: CompressedPayload) -> NDArray[np.uint8]:
    """Descomprime un CompressedPayload codificado con RLE recuperando los bytes exactos."""
    if compressed.codec != "rle":
        raise ValueError(f"Códec inválido para decompress_rle: '{compressed.codec}'.")
    expected = int(compressed.metadata.get("originalByteLength", compressed.original_bytes))
    return decode_packbits_bytes(compressed.payload, expected_length=expected)


def compress_rle_bits(
    bits: NDArray[np.uint8],
    metadata: Optional[Dict[str, Any]] = None,
) -> CompressedPayload:
    """Empaqueta una traza binaria (0/1) a bytes MSB-first y la comprime sin pérdida con RLE."""
    flat_bits = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    total_bits = int(flat_bits.size)
    packed = np.packbits(flat_bits, bitorder="big") if total_bits > 0 else np.zeros(0, dtype=np.uint8)

    meta: Dict[str, Any] = dict(metadata) if metadata else {}
    meta["originalBitLength"] = total_bits
    return compress_rle(packed, source_type="bitstream", metadata=meta)


def decompress_rle_bits(compressed: CompressedPayload) -> NDArray[np.uint8]:
    """Descomprime un CompressedPayload de traza binaria RLE manteniendo longitud exacta de bits."""
    recovered_bytes = decompress_rle(compressed)
    total_bits = compressed.metadata.get("originalBitLength")
    if recovered_bytes.size == 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.unpackbits(recovered_bytes, bitorder="big").astype(np.uint8)
    if total_bits is not None:
        return bits[: int(total_bits)]
    return bits


def compress_rle_image(
    image: ImageMatrix,
    planar: bool = True,
) -> CompressedPayload:
    """
    Comprime una ImageMatrix sin pérdida mediante RLE PackBits.
    Con planar=True separa los canales espaciales (R|G|B|A) para maximizar corridas idénticas.
    """
    total_px = image.width * image.height
    raw = np.asarray(image.pixels[: total_px * image.channels], dtype=np.uint8)
    use_planar = bool(planar and image.channels > 1 and total_px > 0)

    if use_planar:
        ordered = raw.reshape(total_px, image.channels).T.reshape(-1)
    else:
        ordered = raw.reshape(-1)

    return compress_rle(
        ordered,
        source_type="image",
        metadata={
            "width": int(image.width),
            "height": int(image.height),
            "channels": int(image.channels),
            "bitDepth": 8,
            "planar": use_planar,
            "label": image.label,
        },
    )


def decompress_rle_image(compressed: CompressedPayload) -> ImageMatrix:
    """Reconstruye exactamente una ImageMatrix desde un CompressedPayload RLE (MSE = 0)."""
    width = int(compressed.metadata["width"])
    height = int(compressed.metadata["height"])
    channels = int(compressed.metadata["channels"])
    if channels not in (1, 3, 4):
        raise ValueError(f"Número de canales inválido en metadatos RLE: {channels}")

    raw = decompress_rle(compressed)
    total_px = width * height
    expected_len = total_px * channels
    if raw.size != expected_len:
        raise ValueError(
            f"Tamaño reconstruido ({raw.size}) no coincide con {width}x{height}x{channels} ({expected_len})."
        )

    use_planar = bool(compressed.metadata.get("planar", False))
    if use_planar and channels > 1 and total_px > 0:
        pixels = raw.reshape(channels, total_px).T.reshape(-1)
    else:
        pixels = raw.reshape(-1)

    return ImageMatrix(
        width=width,
        height=height,
        channels=channels,  # type: ignore[arg-type]
        pixels=pixels.astype(np.uint8),
        label=str(compressed.metadata.get("label", f"RLE-{width}x{height}")),
    )
