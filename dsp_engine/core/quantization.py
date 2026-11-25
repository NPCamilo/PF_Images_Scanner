"""Cuantización Uniforme PCM y Empaquetado [Byte / Bit]."""

import numpy as np

from dsp_engine.contracts import QuantizationResult, Signal1D


def pack_bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """Empaqueta un arreglo plano de bits (0/1) en bytes (MSB primero)."""
    flat = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    return np.packbits(flat, bitorder="big")


def unpack_bytes_to_bits(packed_bytes: np.ndarray, total_bits: int | None = None) -> np.ndarray:
    """Desempaqueta bytes a un arreglo plano de bits (0/1)."""
    raw = np.asarray(packed_bytes, dtype=np.uint8).reshape(-1)
    bits = np.unpackbits(raw, bitorder="big")
    if total_bits is not None:
        return bits[:total_bits].astype(np.uint8)
    return bits.astype(np.uint8)


def quantize_pcm(
    signal: Signal1D,
    bit_depth: int = 8,
    v_min: float = -1.0,
    v_max: float = 1.0,
) -> QuantizationResult:
    """
    Aplica cuantización uniforme PCM con resolución configurable N ∈ [1, 16] bits/muestra.
    Calcula Δ, error e[n], SQNR teórico (6.02*N + 1.76 dB) y SQNR medido.
    """
    if not isinstance(bit_depth, int) or bit_depth < 1 or bit_depth > 16:
        raise ValueError(f"Resolución PCM inválida ({bit_depth}). Debe estar entre 1 y 16 bits.")
    if v_max <= v_min:
        raise ValueError("v_max debe ser estrictamente mayor que v_min.")

    x = np.clip(signal.samples, v_min, v_max).astype(np.float64)
    num_levels = 1 << bit_depth
    step_delta = (v_max - v_min) / (num_levels - 1)

    levels = np.round((x - v_min) / step_delta).astype(np.int32)
    levels = np.clip(levels, 0, num_levels - 1)

    quantized = (v_min + levels * step_delta).astype(np.float64)
    error_signal = (x - quantized).astype(np.float64)

    power_sig = float(np.mean(x ** 2)) if len(x) > 0 else 0.0
    power_err = float(np.mean(error_signal ** 2)) if len(x) > 0 else 0.0

    sqnr_theoretical = 6.02 * bit_depth + 1.76
    sqnr_measured = (
        float(10.0 * np.log10(max(power_sig, 1e-12) / power_err))
        if power_err > 1e-12
        else 99.9
    )

    # Serialización vectorial MSB -> LSB
    shifts = np.arange(bit_depth - 1, -1, -1, dtype=np.int32)
    bit_matrix = ((levels[:, None] >> shifts) & 1).astype(np.uint8)
    bitstream = bit_matrix.reshape(-1)
    packed_bytes = pack_bits_to_bytes(bitstream)

    return QuantizationResult(
        quantized=quantized,
        levels=levels,
        error_signal=error_signal,
        bit_depth=bit_depth,
        num_levels=num_levels,
        step_size_delta=float(step_delta),
        v_min=float(v_min),
        v_max=float(v_max),
        bitstream=bitstream,
        packed_bytes=packed_bytes,
        sqnr_theoretical_db=float(sqnr_theoretical),
        sqnr_measured_db=sqnr_measured,
    )


def dequantize_bitstream(
    bitstream: np.ndarray,
    bit_depth: int,
    v_min: float = -1.0,
    v_max: float = 1.0,
) -> np.ndarray:
    """Reconstruye muestras cuantizadas PCM a partir de un flujo binario."""
    bits = np.asarray(bitstream, dtype=np.uint8).reshape(-1)
    num_samples = len(bits) // bit_depth
    if num_samples == 0:
        return np.zeros(0, dtype=np.float64)

    bit_matrix = (bits[: num_samples * bit_depth] & 1).reshape(num_samples, bit_depth)
    weights = (1 << np.arange(bit_depth - 1, -1, -1, dtype=np.int32)).astype(np.int32)
    levels = bit_matrix @ weights

    num_levels = 1 << bit_depth
    step_delta = (v_max - v_min) / (num_levels - 1)
    return (v_min + levels * step_delta).astype(np.float64)
