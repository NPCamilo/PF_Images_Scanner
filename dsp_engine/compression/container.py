"""
Empaquetador Binario .dspbin y Exportadores Multi-Formato (.wav, .png, .svg, .bin, .json).
Responsable: Mariano Orrego (feat/compression-docs-mariano)

Especificación de Cabecera Binaria .dspbin (42 bytes Big-Endian):
- magic (4B uint32):          0x44535031 ('DSP1')
- version (1B uint8):         1
- codec_id (1B uint8):        0=none, 1=rle, 2=huffman, 3=mu-law, 4=dct-8x8
- modulation_id (1B uint8):   0=BB, 1=AM-LC, 2=AM-SC, 3=FM, 4=ASK, 5=FSK, 6=BPSK, 7=QPSK
- source_id (1B uint8):       0=audio, 1=image, 2=bitstream
- width (4B uint32):          Ancho de imagen (0 si no aplica)
- height (4B uint32):         Alto de imagen (0 si no aplica)
- channels (1B uint8):        Canales de imagen (0 si no aplica)
- bit_depth (1B uint8):       Resolución en bits (PCM o imagen)
- fs (8B float64):            Frecuencia de muestreo Fs (Hz)
- original_bytes (4B uint32): Tamaño original sin comprimir
- compressed_bytes (4B uint32): Longitud del payload comprimido
- metadata_len (4B uint32):   Longitud del bloque JSON de metadatos extendidos
- crc32 (4B uint32):          Checksum CRC-32 sobre (metadata_json + payload)
Seguido inmediatamente por [metadata_json] + [payload].
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import struct
from typing import Any, Dict, Literal, Optional, Tuple
import zlib
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import (
    AuditMetrics,
    CompressedPayload,
    CompressionType,
    ImageMatrix,
    ModulationType,
)

DSPBIN_MAGIC = 0x44535031  # ASCII 'DSP1'
DSPBIN_VERSION = 1
_HEADER_STRUCT = struct.Struct(">IBBBBIIBBdIIII")  # 42 bytes

_CODEC_TO_ID: Dict[CompressionType, int] = {
    "none": 0,
    "rle": 1,
    "huffman": 2,
    "mu-law": 3,
    "dct-8x8": 4,
}
_ID_TO_CODEC: Dict[int, CompressionType] = {v: k for k, v in _CODEC_TO_ID.items()}

_MOD_TO_ID: Dict[ModulationType, int] = {
    "BB": 0,
    "AM-LC": 1,
    "AM-SC": 2,
    "FM": 3,
    "ASK": 4,
    "FSK": 5,
    "BPSK": 6,
    "QPSK": 7,
}
_ID_TO_MOD: Dict[int, ModulationType] = {v: k for k, v in _MOD_TO_ID.items()}

_SOURCE_TO_ID: Dict[str, int] = {"audio": 0, "image": 1, "bitstream": 2}
_ID_TO_SOURCE: Dict[int, Literal["audio", "image", "bitstream"]] = {
    0: "audio",
    1: "image",
    2: "bitstream",
}


@dataclass(frozen=True)
class DspBinPacket:
    """Contenedor binario .dspbin decodificado con verificación de integridad."""

    magic: int
    version: int
    compressed: CompressedPayload
    modulation: ModulationType
    fs: float
    crc32: int


def pack_dspbin(
    compressed: CompressedPayload,
    modulation: ModulationType = "BB",
    fs: float = 0.0,
) -> bytes:
    """Serializa un CompressedPayload en formato binario propio .dspbin con cabecera 0x44535031 y CRC32."""
    if compressed.codec not in _CODEC_TO_ID:
        raise ValueError(f"Códec no soportado en .dspbin: '{compressed.codec}'.")
    if modulation not in _MOD_TO_ID:
        raise ValueError(f"Esquema de modulación no soportado en .dspbin: '{modulation}'.")

    meta = dict(compressed.metadata)
    source_type = str(meta.get("sourceType", "bitstream"))
    if source_type not in _SOURCE_TO_ID:
        raise ValueError(f"Tipo de fuente inválido para .dspbin: '{source_type}'.")

    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)
    channels = int(meta.get("channels", 0) or 0)
    bit_depth = int(meta.get("bitDepth", 8) or 8)
    effective_fs = float(fs if fs > 0.0 else (meta.get("fs", 0.0) or 0.0))
    if effective_fs > 0.0:
        meta["fs"] = effective_fs

    meta_json_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_bytes = np.asarray(compressed.payload, dtype=np.uint8).tobytes()

    crc32_val = zlib.crc32(meta_json_bytes + payload_bytes) & 0xFFFFFFFF

    header = _HEADER_STRUCT.pack(
        DSPBIN_MAGIC,
        DSPBIN_VERSION,
        _CODEC_TO_ID[compressed.codec],
        _MOD_TO_ID[modulation],
        _SOURCE_TO_ID[source_type],
        width,
        height,
        channels,
        bit_depth,
        effective_fs,
        int(compressed.original_bytes),
        len(payload_bytes),
        len(meta_json_bytes),
        crc32_val,
    )
    return header + meta_json_bytes + payload_bytes


def unpack_dspbin(raw_bytes: bytes) -> DspBinPacket:
    """Deserializa y valida cabecera mágica (0x44535031) y CRC32 de un archivo .dspbin."""
    if len(raw_bytes) < _HEADER_STRUCT.size:
        raise ValueError("Buffer .dspbin demasiado corto para contener cabecera válida.")

    (
        magic,
        version,
        codec_id,
        mod_id,
        source_id,
        width,
        height,
        channels,
        bit_depth,
        fs,
        orig_bytes,
        comp_bytes,
        meta_len,
        expected_crc,
    ) = _HEADER_STRUCT.unpack_from(raw_bytes, 0)

    if magic != DSPBIN_MAGIC:
        raise ValueError(f"Cabecera mágica .dspbin inválida: 0x{magic:08X} (esperada 0x{DSPBIN_MAGIC:08X}).")
    if version != DSPBIN_VERSION:
        raise ValueError(f"Versión .dspbin no soportada: {version}.")
    if codec_id not in _ID_TO_CODEC:
        raise ValueError(f"Identificador de códec desconocido: {codec_id}.")
    if mod_id not in _ID_TO_MOD:
        raise ValueError(f"Identificador de modulación desconocido: {mod_id}.")
    if source_id not in _ID_TO_SOURCE:
        raise ValueError(f"Identificador de fuente desconocido: {source_id}.")

    expected_total = _HEADER_STRUCT.size + meta_len + comp_bytes
    if len(raw_bytes) < expected_total:
        raise ValueError(
            f"Buffer .dspbin truncado: {len(raw_bytes)} bytes recibidos, {expected_total} esperados."
        )

    meta_start = _HEADER_STRUCT.size
    payload_start = meta_start + meta_len
    meta_bytes = raw_bytes[meta_start:payload_start]
    payload_bytes = raw_bytes[payload_start : payload_start + comp_bytes]

    actual_crc = zlib.crc32(meta_bytes + payload_bytes) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            f"Error de integridad CRC32 en .dspbin (esperado 0x{expected_crc:08X}, calculado 0x{actual_crc:08X})."
        )

    meta: Dict[str, Any] = json.loads(meta_bytes.decode("utf-8")) if meta_len > 0 else {}
    meta.setdefault("sourceType", _ID_TO_SOURCE[source_id])
    if width > 0:
        meta.setdefault("width", width)
    if height > 0:
        meta.setdefault("height", height)
    if channels > 0:
        meta.setdefault("channels", channels)
    meta.setdefault("bitDepth", bit_depth)
    if fs > 0.0:
        meta.setdefault("fs", fs)

    payload_arr = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
    ratio = float(orig_bytes / comp_bytes) if comp_bytes > 0 else (1.0 if orig_bytes == 0 else float(orig_bytes))
    savings = float((1.0 - (comp_bytes / orig_bytes)) * 100.0) if orig_bytes > 0 else 0.0

    compressed = CompressedPayload(
        codec=_ID_TO_CODEC[codec_id],
        original_bytes=orig_bytes,
        compressed_bytes=comp_bytes,
        compression_ratio=ratio,
        savings_percent=savings,
        payload=payload_arr,
        metadata=meta,
    )

    return DspBinPacket(
        magic=magic,
        version=version,
        compressed=compressed,
        modulation=_ID_TO_MOD[mod_id],
        fs=fs,
        crc32=actual_crc,
    )


def export_wav_bytes(
    samples: NDArray[np.float64],
    fs: int,
    bit_depth: int = 16,
) -> bytes:
    """
    Exporta muestras de audio normalizadas [-1, 1] a archivo estándar RIFF WAV PCM (8 o 16 bits)
    construyendo explícitamente la cabecera binaria RIFF/WAVE/fmt/data de 44 bytes.
    """
    if fs <= 0:
        raise ValueError(f"La frecuencia de muestreo WAV debe ser positiva (recibida: {fs}).")
    if bit_depth not in (8, 16):
        raise ValueError(f"Profundidad de bits WAV no soportada ({bit_depth}). Use 8 o 16.")

    clipped = np.clip(np.asarray(samples, dtype=np.float64).reshape(-1), -1.0, 1.0)
    num_channels = 1
    bytes_per_sample = bit_depth // 8

    if bit_depth == 8:
        pcm_bytes = np.round((clipped + 1.0) * 127.5).astype(np.uint8).tobytes()
    else:
        pcm_bytes = np.round(clipped * 32767.0).astype("<i2").tobytes()

    data_size = len(pcm_bytes)
    byte_rate = int(fs) * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,              # Subchunk1Size para PCM
        1,               # AudioFormat = 1 (PCM lineal)
        num_channels,
        int(fs),
        byte_rate,
        block_align,
        bit_depth,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def import_wav_bytes(wav_bytes: bytes) -> Tuple[NDArray[np.float64], int, int]:
    """Lee un buffer RIFF WAV PCM estándar retornando (samples_norm, fs, bit_depth)."""
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Buffer inválido: no contiene cabecera RIFF/WAVE.")

    offset = 12
    fs = 0
    bit_depth = 0
    num_channels = 1
    pcm_data = b""

    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(wav_bytes):
            raise ValueError("Chunk WAV truncado.")
        if chunk_id == b"fmt ":
            _, num_channels, fs, _, _, bit_depth = struct.unpack_from("<HHIIHH", wav_bytes, chunk_start)
        elif chunk_id == b"data":
            pcm_data = wav_bytes[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size & 1)

    if fs <= 0 or bit_depth not in (8, 16):
        raise ValueError("Parámetros WAV no soportados o faltantes.")

    if bit_depth == 8:
        arr = (np.frombuffer(pcm_data, dtype=np.uint8).astype(np.float64) - 127.5) / 127.5
    else:
        arr = np.frombuffer(pcm_data, dtype="<i2").astype(np.float64) / 32767.0

    if num_channels > 1:
        arr = arr.reshape(-1, num_channels).mean(axis=1)

    return arr.astype(np.float64), int(fs), int(bit_depth)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Construye un chunk PNG con longitud, tipo, datos y CRC32."""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def export_png_bytes(image: ImageMatrix) -> bytes:
    """
    Codifica una ImageMatrix (1, 3 o 4 canales) a PNG estándar sin depender de Pillow/C.
    100% compatible con Pyodide WASM.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Las dimensiones de imagen PNG deben ser positivas.")
    if image.channels not in (1, 3, 4):
        raise ValueError(f"Canales PNG inválidos: {image.channels}")

    expected_len = image.width * image.height * image.channels
    if image.pixels.size < expected_len:
        raise ValueError("Buffer de píxeles insuficiente para codificar PNG.")

    color_type_map = {1: 0, 3: 2, 4: 6}  # Grayscale, RGB, RGBA
    color_type = color_type_map[image.channels]

    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, color_type, 0, 0, 0)
    raw = np.asarray(image.pixels[:expected_len], dtype=np.uint8).reshape(
        image.height, image.width * image.channels
    )

    # Prepend filter byte 0 (None) a cada scanline
    scanlines = np.hstack([np.zeros((image.height, 1), dtype=np.uint8), raw])
    idat_payload = zlib.compress(scanlines.tobytes(), level=6)

    signature = b"\x89PNG\r\n\x1a\n"
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat_payload)
        + _png_chunk(b"IEND", b"")
    )


def export_svg_string(image: ImageMatrix, pixel_scale: int = 1) -> str:
    """
    Exporta una ImageMatrix como gráfico vectorial SVG agrupando píxeles horizontales
    idénticos contiguos (RLE espacial) para mantener tamaño reducido.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Las dimensiones de imagen SVG deben ser positivas.")
    scale = max(1, int(pixel_scale))
    w, h, ch = image.width, image.height, image.channels
    arr = np.asarray(image.pixels[: w * h * ch], dtype=np.uint8).reshape(h, w, ch)

    rects: list[str] = []
    for y in range(h):
        x = 0
        while x < w:
            px = arr[y, x]
            if ch == 1:
                v = int(px[0])
                color = f"#{v:02x}{v:02x}{v:02x}"
                alpha = None
            elif ch == 3:
                color = f"#{int(px[0]):02x}{int(px[1]):02x}{int(px[2]):02x}"
                alpha = None
            else:
                color = f"#{int(px[0]):02x}{int(px[1]):02x}{int(px[2]):02x}"
                a_val = int(px[3])
                alpha = None if a_val == 255 else round(a_val / 255.0, 3)

            run = 1
            while x + run < w and np.array_equal(arr[y, x + run], px):
                run += 1

            opacity_attr = f' fill-opacity="{alpha}"' if alpha is not None else ""
            rects.append(
                f'<rect x="{x * scale}" y="{y * scale}" width="{run * scale}" '
                f'height="{scale}" fill="{color}"{opacity_attr}/>'
            )
            x += run

    body = "\n  ".join(rects)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * scale}" height="{h * scale}" '
        f'viewBox="0 0 {w * scale} {h * scale}" shape-rendering="crispEdges">\n'
        f"  {body}\n"
        f"</svg>"
    )


def export_bin_bytes(data: NDArray[np.uint8], is_bitstream: bool = False) -> bytes:
    """Exporta una traza de bits (empaquetada MSB-first) o bytes crudos a formato .bin."""
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return b""
    if is_bitstream:
        return np.packbits(arr & 1, bitorder="big").tobytes()
    return arr.tobytes()


def export_session_report_json(
    compressed: CompressedPayload,
    modulation: Optional[ModulationType] = None,
    fs: Optional[float] = None,
    metrics: Optional[AuditMetrics] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Genera un reporte estructurado de sesión en JSON con compresión, modulación y auditoría."""
    report: Dict[str, Any] = {
        "schemaVersion": "1.0",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "compression": {
            "codec": compressed.codec,
            "originalBytes": int(compressed.original_bytes),
            "compressedBytes": int(compressed.compressed_bytes),
            "compressionRatio": float(compressed.compression_ratio),
            "savingsPercent": float(compressed.savings_percent),
            "metadata": compressed.metadata,
        },
        "transmission": {
            "modulation": modulation or "BB",
            "fsHz": float(fs if fs is not None else compressed.metadata.get("fs", 0.0) or 0.0),
        },
    }

    if metrics is not None:
        report["auditMetrics"] = {
            "mse": float(metrics.mse),
            "rmse": float(metrics.rmse),
            "snrDb": float(metrics.snr_db),
            "psnrDb": float(metrics.psnr_db),
            "ber": float(metrics.ber),
            "bitErrors": int(metrics.bit_errors),
            "totalBits": int(metrics.total_bits),
            "pearsonCorrelation": float(metrics.pearson_correlation),
        }

    if extra:
        report["extra"] = extra

    return json.dumps(report, indent=2, ensure_ascii=False)
