"""
Compresión Logarítmica con Pérdida Mu-Law (ITU-T G.711, μ = 255) para Audio y Señales 1D.
Responsable: Mariano Orrego (feat/compression-docs-mariano)

Ecuaciones de Compansión No Lineal (x ∈ [-1, 1], μ = 255):
- Compresión (Compander):
      F(x) = sgn(x) * ln(1 + μ * |x|) / ln(1 + μ)
- Expansión (Expander):
      F^{-1}(y) = sgn(y) * ((1 + μ)^{|y|} - 1) / μ

Ventaja Matemática: Incrementa la resolución efectiva en amplitudes bajas (voz/audio),
logrando SQNR casi constante sobre un amplio rango dinámico respecto a PCM uniforme.
"""

import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import CompressedPayload, Signal1D


def mu_law_compand(samples: NDArray[np.float64], mu: float = 255.0) -> NDArray[np.float64]:
    """
    Aplica la ley de compresión logarítmica μ-law continua sobre muestras normalizadas [-1, 1].
    F(x) = sgn(x) * ln(1 + μ|x|) / ln(1 + μ).
    """
    if mu <= 0.0:
        raise ValueError(f"El parámetro μ debe ser positivo (recibido: {mu}).")
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    return (np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)).astype(np.float64)


def mu_law_expand(companded: NDArray[np.float64], mu: float = 255.0) -> NDArray[np.float64]:
    """
    Aplica la ley de expansión inversa μ-law sobre muestras comprimidas [-1, 1].
    F^{-1}(y) = sgn(y) * ((1 + μ)^{|y|} - 1) / μ.
    """
    if mu <= 0.0:
        raise ValueError(f"El parámetro μ debe ser positivo (recibido: {mu}).")
    y = np.clip(np.asarray(companded, dtype=np.float64), -1.0, 1.0)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    return (np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)).astype(np.float64)


def compress_mu_law(
    signal: Signal1D,
    mu: float = 255.0,
    compressed_bit_depth: int = 8,
    original_bit_depth: int = 16,
) -> CompressedPayload:
    """
    Comprime una Signal1D mediante companding μ-law (μ = 255) y cuantización uniforme de
    `compressed_bit_depth` bits (por defecto 8 bits frente a referencia PCM de 16 bits -> CR = 2.0).
    """
    if not isinstance(compressed_bit_depth, int) or compressed_bit_depth < 2 or compressed_bit_depth > 16:
        raise ValueError(
            f"Resolución comprimida μ-law inválida ({compressed_bit_depth}). Debe estar entre 2 y 16 bits."
        )
    if not isinstance(original_bit_depth, int) or original_bit_depth < 1 or original_bit_depth > 32:
        raise ValueError(f"Resolución original inválida ({original_bit_depth}).")

    x = np.asarray(signal.samples, dtype=np.float64)
    n_samples = int(x.size)
    original_bytes = int(np.ceil((n_samples * original_bit_depth) / 8.0))

    if n_samples == 0:
        return CompressedPayload(
            codec="mu-law",
            original_bytes=0,
            compressed_bytes=0,
            compression_ratio=1.0,
            savings_percent=0.0,
            payload=np.zeros(0, dtype=np.uint8),
            metadata={
                "sourceType": "audio",
                "fs": float(signal.fs),
                "bitDepth": compressed_bit_depth,
                "originalBitDepth": original_bit_depth,
                "mu": float(mu),
                "sampleCount": 0,
                "peakScale": 1.0,
                "label": signal.label,
            },
        )

    # Normalización adaptativa al pico si la señal excede [-1, 1]
    max_abs = float(np.max(np.abs(x)))
    peak_scale = max_abs if max_abs > 1.0 else 1.0
    x_norm = x / peak_scale

    y = mu_law_compand(x_norm, mu=mu)

    # Cuantizador simétrico mid-tread con cero exacto: niveles en [-q_max, +q_max]
    q_max = (1 << (compressed_bit_depth - 1)) - 1
    q_signed = np.clip(np.round(y * q_max), -q_max, q_max).astype(np.int32)
    q_unsigned = (q_signed + (1 << (compressed_bit_depth - 1))).astype(np.uint32)

    if compressed_bit_depth <= 8:
        if compressed_bit_depth == 8:
            payload = q_unsigned.astype(np.uint8)
        else:
            # Empaquetado compacto bit a bit para resoluciones < 8 bits
            shifts = np.arange(compressed_bit_depth - 1, -1, -1, dtype=np.int32)
            bits = ((q_unsigned[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
            payload = np.packbits(bits, bitorder="big")
    else:
        # 9..16 bits: empaquetado de 2 bytes big-endian por muestra
        payload = np.frombuffer(q_unsigned.astype(">u2").tobytes(), dtype=np.uint8).copy()

    compressed_bytes = int(payload.size)
    ratio = float(original_bytes / compressed_bytes) if compressed_bytes > 0 else 1.0
    savings = float((1.0 - (compressed_bytes / original_bytes)) * 100.0) if original_bytes > 0 else 0.0

    # Estimación rápida de SNR de compresión sobre la señal
    y_q = q_signed.astype(np.float64) / float(q_max)
    x_rec = mu_law_expand(y_q, mu=mu) * peak_scale
    sig_power = float(np.mean(x ** 2))
    err_power = float(np.mean((x - x_rec) ** 2))
    snr_db = float(10.0 * np.log10(max(sig_power, 1e-12) / err_power)) if err_power > 1e-15 else 99.9

    return CompressedPayload(
        codec="mu-law",
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        savings_percent=savings,
        payload=payload,
        metadata={
            "sourceType": "audio",
            "fs": float(signal.fs),
            "bitDepth": compressed_bit_depth,
            "originalBitDepth": original_bit_depth,
            "mu": float(mu),
            "sampleCount": n_samples,
            "peakScale": peak_scale,
            "snrDb": snr_db,
            "label": signal.label,
        },
    )


def decompress_mu_law(compressed: CompressedPayload) -> Signal1D:
    """Reconstruye una Signal1D expandiendo el flujo cuantizado μ-law."""
    if compressed.codec != "mu-law":
        raise ValueError(f"Códec inválido para decompress_mu_law: '{compressed.codec}'.")

    fs = float(compressed.metadata.get("fs", 8000.0))
    mu = float(compressed.metadata.get("mu", 255.0))
    bit_depth = int(compressed.metadata.get("bitDepth", 8))
    n_samples = int(compressed.metadata.get("sampleCount", 0))
    peak_scale = float(compressed.metadata.get("peakScale", 1.0))
    label = str(compressed.metadata.get("label", "Mu-Law Reconstructed"))

    if n_samples == 0 or compressed.payload.size == 0:
        return Signal1D(
            id="sig-mulaw-dec",
            samples=np.zeros(0, dtype=np.float64),
            fs=fs,
            ts=1.0 / fs if fs > 0 else 0.0,
            duration=0.0,
            is_discrete=True,
            label=label,
        )

    payload = np.asarray(compressed.payload, dtype=np.uint8)
    offset = 1 << (bit_depth - 1)
    q_max = offset - 1

    if bit_depth == 8:
        q_unsigned = payload[:n_samples].astype(np.int32)
    elif bit_depth < 8:
        total_bits = n_samples * bit_depth
        bits = np.unpackbits(payload, bitorder="big")[:total_bits].reshape(n_samples, bit_depth)
        weights = (1 << np.arange(bit_depth - 1, -1, -1, dtype=np.int32)).astype(np.int32)
        q_unsigned = bits @ weights
    else:
        q_unsigned = np.frombuffer(payload[: n_samples * 2].tobytes(), dtype=">u2").astype(np.int32)

    q_signed = q_unsigned - offset
    y_q = np.clip(q_signed.astype(np.float64) / float(q_max), -1.0, 1.0)
    samples = (mu_law_expand(y_q, mu=mu) * peak_scale).astype(np.float64)

    ts = 1.0 / fs if fs > 0 else 0.0
    return Signal1D(
        id="sig-mulaw-dec",
        samples=samples,
        fs=fs,
        ts=ts,
        duration=float(samples.size * ts),
        is_discrete=True,
        label=label,
    )
