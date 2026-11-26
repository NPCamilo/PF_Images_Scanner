"""Reconstrucción DAC mediante interpolación sinc truncada y retención de orden cero."""

import math
from typing import Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import Signal1D
from dsp_engine.demodulation.filters import lowpass_filter


def _validate_rate(fs: float, name: str) -> float:
    """Valida una tasa de muestreo finita y positiva."""
    value = float(fs)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} inválida: {fs} Hz.")
    return value


def _target_length(
    num_samples: int,
    fs_in: float,
    fs_out: float,
    target_length: Optional[int],
) -> int:
    """Deriva la cantidad de muestras de salida cuando no se especifica explícitamente."""
    if target_length is None:
        return int(round(num_samples * fs_out / fs_in))
    if isinstance(target_length, bool) or not isinstance(target_length, int) or target_length < 0:
        raise ValueError(f"target_length inválido: {target_length}.")
    return target_length


def reconstruct_sinc(
    sampled_values: NDArray[np.float64],
    fs_in: float,
    fs_out: float,
    target_length: Optional[int] = None,
    kernel_half_width: int = 16,
) -> NDArray[np.float64]:
    """Interpola xr(t) = sum x[n] sinc((t-n/Fsi)*Fsi) con núcleo sinc truncado y ventana de Hann."""
    x = np.asarray(sampled_values, dtype=np.float64).reshape(-1)
    input_rate = _validate_rate(fs_in, "fs_in")
    output_rate = _validate_rate(fs_out, "fs_out")
    length = _target_length(x.size, input_rate, output_rate, target_length)
    if x.size == 0 or length == 0:
        return np.empty(0, dtype=np.float64)
    if isinstance(kernel_half_width, bool) or not isinstance(kernel_half_width, int) or kernel_half_width < 1:
        raise ValueError("kernel_half_width debe ser entero >= 1.")

    taps = 2 * kernel_half_width
    window = np.hanning(taps) if taps >= 3 else np.ones(taps, dtype=np.float64)
    offsets = np.arange(-kernel_half_width + 1, kernel_half_width + 1, dtype=np.int64)
    chunk_size = max(1, 200000 // taps)
    output = np.zeros(length, dtype=np.float64)

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        positions = np.arange(start, stop, dtype=np.float64) * input_rate / output_rate
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        distances = positions[:, None] - indices.astype(np.float64)
        weights = np.sinc(distances) * window[None, :]
        valid = (indices >= 0) & (indices < x.size)
        values = x[np.clip(indices, 0, x.size - 1)]

        numerator = np.sum(weights * values * valid, axis=1)
        denominator = np.sum(weights * valid, axis=1)
        usable = np.abs(denominator) > 1e-12
        np.divide(numerator, denominator, out=output[start:stop], where=usable)
    return np.asarray(output, dtype=np.float64)


def reconstruct_zoh(
    sampled_values: NDArray[np.float64],
    fs_in: float,
    fs_out: float,
    target_length: Optional[int] = None,
    cutoff_hz: Optional[float] = None,
    filter_order: int = 4,
) -> NDArray[np.float64]:
    """Mantiene x[n] durante Ts centrando el escalón en n/Fsi y suaviza imágenes con Butterworth."""
    x = np.asarray(sampled_values, dtype=np.float64).reshape(-1)
    input_rate = _validate_rate(fs_in, "fs_in")
    output_rate = _validate_rate(fs_out, "fs_out")
    length = _target_length(x.size, input_rate, output_rate, target_length)
    if x.size == 0 or length == 0:
        return np.empty(0, dtype=np.float64)

    positions = np.arange(length, dtype=np.float64) * input_rate / output_rate
    indices = np.clip(np.floor(positions + 0.5).astype(np.int64), 0, x.size - 1)
    held = x[indices]
    cutoff = min(0.45 * input_rate, 0.45 * output_rate) if cutoff_hz is None else float(cutoff_hz)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < output_rate / 2.0:
        raise ValueError(f"Corte de reconstrucción inválido: {cutoff_hz} Hz.")
    return lowpass_filter(held, cutoff, output_rate, filter_order)


def _optional_float_kwarg(kwargs: Dict[str, object], name: str) -> Optional[float]:
    """Lee un argumento opcional convertido a float."""
    value = kwargs.get(name)
    return None if value is None else float(value)


def _int_kwarg(kwargs: Dict[str, object], name: str, default: int) -> int:
    """Lee un argumento entero con valor por defecto."""
    value = kwargs.get(name, default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} debe ser entero.")
    return int(value)


def reconstruct_dac(
    sampled_values: NDArray[np.float64],
    fs_in: float,
    fs_out: float,
    method: Literal["sinc", "zoh"] = "sinc",
    target_length: Optional[int] = None,
    **kwargs: object,
) -> Signal1D:
    """Devuelve la señal reconstruida como Signal1D continua a la nueva tasa."""
    method_value = str(method)
    if method_value not in ("sinc", "zoh"):
        raise ValueError(f"Método de reconstrucción no soportado: {method}")

    if method_value == "sinc":
        width = _int_kwarg(kwargs, "kernel_half_width", 16)
        samples = reconstruct_sinc(
            sampled_values,
            fs_in,
            fs_out,
            target_length=target_length,
            kernel_half_width=width,
        )
    else:
        samples = reconstruct_zoh(
            sampled_values,
            fs_in,
            fs_out,
            target_length=target_length,
            cutoff_hz=_optional_float_kwarg(kwargs, "cutoff_hz"),
            filter_order=_int_kwarg(kwargs, "filter_order", 4),
        )

    output_rate = _validate_rate(fs_out, "fs_out")
    return Signal1D(
        id=f"dac-{method_value}",
        samples=np.asarray(samples, dtype=np.float64),
        fs=output_rate,
        ts=1.0 / output_rate,
        duration=float(samples.size / output_rate),
        is_discrete=False,
        label=f"DAC {method_value.upper()} @ {output_rate:g} Hz",
        metadata={
            "method": method_value,
            "source_fs": float(fs_in),
            "target_length": int(samples.size),
        },
    )
