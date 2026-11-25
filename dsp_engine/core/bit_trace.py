"""Utilidades de Traza de Bits y Codificación de Línea en Banda Base (BB)."""

import numpy as np

from dsp_engine.contracts import LineCodingType


def generate_bit_trace_waveform(
    bits: np.ndarray,
    samples_per_bit: int = 16,
    coding: LineCodingType = "polar-nrz",
) -> np.ndarray:
    """
    Convierte una secuencia binaria (0/1) en forma de onda temporal en Banda Base (BB).
    - unipolar-nrz: 1 -> +1.0, 0 -> 0.0
    - polar-nrz:    1 -> +1.0, 0 -> -1.0
    - manchester:   1 -> [+1, -1], 0 -> [-1, +1] (transición en mitad de bit, DC = 0)
    """
    flat = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    spb = max(2, int(samples_per_bit))
    if len(flat) == 0:
        return np.zeros(0, dtype=np.float64)

    if coding == "unipolar-nrz":
        levels = np.where(flat == 1, 1.0, 0.0)
        return np.repeat(levels, spb).astype(np.float64)

    if coding == "polar-nrz":
        levels = np.where(flat == 1, 1.0, -1.0)
        return np.repeat(levels, spb).astype(np.float64)

    if coding == "manchester":
        half = spb // 2
        rest = spb - half
        out = np.zeros(len(flat) * spb, dtype=np.float64)
        for i, b in enumerate(flat):
            base = i * spb
            if b == 1:
                out[base : base + half] = 1.0
                out[base + half : base + spb] = -1.0
            else:
                out[base : base + half] = -1.0
                out[base + half : base + spb] = 1.0
            _ = rest
        return out

    raise ValueError(f"Codificación de línea no soportada: {coding}")


def text_to_bits(text: str) -> np.ndarray:
    """Codifica una cadena UTF-8 en arreglo plano de bits (MSB primero)."""
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="big").astype(np.uint8)


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convierte una cadena hexadecimal limpia en arreglo plano de bits."""
    cleaned = "".join(c for c in hex_str.strip() if c in "0123456789abcdefABCDEF")
    if len(cleaned) == 0:
        return np.zeros(0, dtype=np.uint8)
    if len(cleaned) % 2 == 1:
        cleaned = "0" + cleaned
    raw = np.frombuffer(bytes.fromhex(cleaned), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="big").astype(np.uint8)


def bits_to_hex(bits: np.ndarray, max_bytes: int = 64) -> str:
    """Formatea los primeros N bytes de un bitstream en representación hexadecimal legible."""
    flat = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    packed = np.packbits(flat, bitorder="big")[:max_bytes]
    return " ".join(f"{b:02X}" for b in packed)
