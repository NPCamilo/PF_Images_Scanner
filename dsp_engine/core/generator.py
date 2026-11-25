"""Generación sintética de señales x(t)/x[n], secuencias PRBS e imágenes de prueba."""

import math
from typing import Literal, Optional
import numpy as np

from dsp_engine.contracts import ImageMatrix, Signal1D, SyntheticWaveformType


def generate_synthetic_signal(
    waveform: SyntheticWaveformType = "sine",
    amplitude: float = 1.0,
    frequency_hz: float = 440.0,
    secondary_frequency_hz: Optional[float] = None,
    phase_rad: float = 0.0,
    dc_offset: float = 0.0,
    duration_sec: float = 0.5,
    fs: float = 8000.0,
    duty_cycle: float = 0.5,
) -> Signal1D:
    """Genera una señal sintética parametrizable x(t) / x[n] usando NumPy vectorizado."""
    if fs <= 0 or not math.isfinite(fs):
        raise ValueError(f"Frecuencia de muestreo inválida: {fs} Hz.")
    if duration_sec <= 0 or not math.isfinite(duration_sec):
        raise ValueError(f"Duración inválida: {duration_sec} s.")

    n = max(1, round(duration_sec * fs))
    ts = 1.0 / fs
    t = np.arange(n, dtype=np.float64) * ts
    f2 = secondary_frequency_hz if secondary_frequency_hz is not None else frequency_hz * 2.5
    duty = min(0.95, max(0.05, duty_cycle))
    angle = 2.0 * np.pi * frequency_hz * t + phase_rad

    if waveform == "sine":
        val = np.sin(angle)
    elif waveform == "cosine":
        val = np.cos(angle)
    elif waveform == "square":
        cycle_pos = np.mod(frequency_hz * t + phase_rad / (2.0 * np.pi), 1.0)
        val = np.where(cycle_pos < duty, 1.0, -1.0)
    elif waveform == "pulse-train":
        cycle_pos = np.mod(frequency_hz * t + phase_rad / (2.0 * np.pi), 1.0)
        val = np.where(cycle_pos < duty, 1.0, 0.0)
    elif waveform == "triangle":
        cycle_pos = np.mod(frequency_hz * t + phase_rad / (2.0 * np.pi), 1.0)
        val = 4.0 * np.abs(cycle_pos - 0.5) - 1.0
    elif waveform == "sawtooth":
        cycle_pos = np.mod(frequency_hz * t + phase_rad / (2.0 * np.pi), 1.0)
        val = 2.0 * cycle_pos - 1.0
    elif waveform == "chirp":
        k = (f2 - frequency_hz) / duration_sec
        inst_phase = 2.0 * np.pi * (frequency_hz * t + 0.5 * k * (t ** 2)) + phase_rad
        val = np.sin(inst_phase)
    elif waveform == "multitone":
        val = 0.6 * np.sin(angle) + 0.4 * np.sin(2.0 * np.pi * f2 * t + phase_rad)
    else:
        raise ValueError(f"Forma de onda no soportada: {waveform}")

    samples = (dc_offset + amplitude * val).astype(np.float64)
    return Signal1D(
        id=f"sig-{waveform}",
        samples=samples,
        fs=float(fs),
        ts=float(ts),
        duration=float(n * ts),
        is_discrete=False,
        label=f"{waveform.upper()} ({frequency_hz} Hz @ {fs} Hz)",
        metadata={
            "waveform": waveform,
            "frequency_hz": frequency_hz,
            "amplitude": amplitude,
            "dc_offset": dc_offset,
        },
    )


def generate_prbs(length: int, seed: int = 0xACE1) -> np.ndarray:
    """Genera una Secuencia Binaria Pseudoaleatoria (PRBS) determinística mediante LFSR de 16 bits."""
    if length <= 0:
        raise ValueError("La longitud de la secuencia PRBS debe ser mayor que 0.")

    bits = np.zeros(length, dtype=np.uint8)
    lfsr = (seed & 0xFFFF) or 0xACE1
    for i in range(length):
        bits[i] = lfsr & 1
        feedback = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
        lfsr = (lfsr >> 1) | (feedback << 15)
    return bits


def generate_test_image(
    width: int = 64,
    height: int = 64,
    pattern: Literal["bars", "checkerboard", "gradient", "target"] = "bars",
) -> ImageMatrix:
    """Genera imágenes patrón RGBA sintéticas para calibración y pruebas sin archivos externos."""
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255

    if pattern == "bars":
        palette = np.array(
            [
                [235, 235, 235],
                [235, 200, 40],
                [40, 210, 220],
                [60, 210, 90],
                [220, 60, 200],
                [225, 50, 50],
            ],
            dtype=np.uint8,
        )
        for x in range(width):
            band = min(5, int((x / max(1, width)) * 6))
            pixels[:, x, :3] = palette[band]
    elif pattern == "checkerboard":
        cell = max(4, width // 8)
        yy, xx = np.indices((height, width))
        mask = ((xx // cell) + (yy // cell)) % 2 == 0
        val = np.where(mask, 240, 20).astype(np.uint8)
        pixels[:, :, 0] = val
        pixels[:, :, 1] = val
        pixels[:, :, 2] = val
    elif pattern == "gradient":
        x_grad = np.linspace(0, 255, width, dtype=np.uint8)
        y_grad = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        pixels[:, :, 0] = x_grad
        pixels[:, :, 1] = y_grad
        pixels[:, :, 2] = 140
    elif pattern == "target":
        yy, xx = np.indices((height, width))
        dx = xx - width / 2.0
        dy = yy - height / 2.0
        dist = np.sqrt(dx ** 2 + dy ** 2)
        ring = np.where(np.sin(dist * 0.35) > 0, 220, 30).astype(np.uint8)
        pixels[:, :, 0] = ring
        pixels[:, :, 1] = ring
        pixels[:, :, 2] = ring
        cross = (np.abs(dx) < 3) | (np.abs(dy) < 3)
        pixels[cross, 0] = 230
        pixels[cross, 1] = 40
        pixels[cross, 2] = 40
    else:
        raise ValueError(f"Patrón de imagen no soportado: {pattern}")

    return ImageMatrix(
        width=width,
        height=height,
        channels=4,
        pixels=pixels.reshape(-1),
        label=f"Pattern-{pattern}-{width}x{height}",
    )
