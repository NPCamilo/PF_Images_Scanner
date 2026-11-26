"""Filtros Butterworth en forma SOS con fase cero para el receptor DSP."""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def _validate_fs(fs: float) -> float:
    """Valida una frecuencia de muestreo finita y positiva."""
    value = float(fs)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Frecuencia de muestreo inválida: {fs} Hz.")
    return value


def _validate_order(order: int) -> int:
    """Valida el orden del filtro como entero positivo."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"Orden de filtro inválido: {order}. Debe ser entero >= 1.")
    return order


def design_butterworth_lpf(
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Diseña un pasabajos Butterworth: H(s) = 1 / sqrt(1 + (f/fc)^(2N))."""
    fs_value = _validate_fs(fs)
    cutoff = float(cutoff_hz)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < fs_value / 2.0:
        raise ValueError(f"Corte inválido: {cutoff_hz} Hz. Debe estar en (0, Fs/2).")
    order_value = _validate_order(order)
    sos = signal.butter(order_value, cutoff, btype="lowpass", fs=fs_value, output="sos")
    return np.asarray(sos, dtype=np.float64)


def design_butterworth_bpf(
    lowcut_hz: float,
    highcut_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Diseña un pasabanda Butterworth entre lowcut y highcut, ambos bajo Nyquist."""
    fs_value = _validate_fs(fs)
    low = float(lowcut_hz)
    high = float(highcut_hz)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Los cortes del pasabanda deben ser finitos.")
    if not 0.0 < low < high < fs_value / 2.0:
        raise ValueError(f"Rango inválido: {lowcut_hz}..{highcut_hz} Hz con Fs={fs_value} Hz.")
    order_value = _validate_order(order)
    sos = signal.butter(
        order_value,
        [low, high],
        btype="bandpass",
        fs=fs_value,
        output="sos",
    )
    return np.asarray(sos, dtype=np.float64)


def apply_zero_phase_filter(
    samples: NDArray[np.float64],
    sos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aplica el filtro SOS en tiempo cero con sosfiltfilt, ajustando padlen a la longitud disponible."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    coefficients = np.asarray(sos, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 6 or coefficients.shape[0] < 1:
        raise ValueError("Los coeficientes SOS deben tener forma (n_sections, 6).")
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    if x.size < 2:
        return np.asarray(signal.sosfilt(coefficients, x), dtype=np.float64)

    default_padlen = 3 * (
        2 * coefficients.shape[0]
        + 1
        - min(int(np.count_nonzero(coefficients[:, 2] == 0.0)), int(np.count_nonzero(coefficients[:, 5] == 0.0)))
    )
    padlen = max(0, min(default_padlen, x.size - 1))
    try:
        filtered = signal.sosfiltfilt(coefficients, x, padlen=padlen)
    except ValueError:
        filtered = signal.sosfilt(coefficients, x)
    return np.asarray(filtered, dtype=np.float64)


def lowpass_filter(
    samples: NDArray[np.float64],
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Filtra x[n] con un pasabajos Butterworth de fase cero."""
    sos = design_butterworth_lpf(cutoff_hz, fs, order)
    return apply_zero_phase_filter(samples, sos)


def bandpass_filter(
    samples: NDArray[np.float64],
    lowcut_hz: float,
    highcut_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Filtra x[n] con un pasabanda Butterworth de fase cero."""
    sos = design_butterworth_bpf(lowcut_hz, highcut_hz, fs, order)
    return apply_zero_phase_filter(samples, sos)
