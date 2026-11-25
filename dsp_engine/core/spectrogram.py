"""Transformada de Fourier de Tiempo Corto (STFT) para Espectrograma 2D (Waterfall)."""

from typing import Dict, Any
import numpy as np

from dsp_engine.contracts import Signal1D, WindowType
from dsp_engine.core.fft import build_window, next_power_of_two


def compute_spectrogram(
    signal: Signal1D,
    window_size: int = 256,
    hop_size: int = 64,
    window: WindowType = "hann",
) -> Dict[str, Any]:
    """
    Calcula la matriz tiempo-frecuencia STFT en dBFS para visualización en cascada.
    Garantiza al menos 1 ventana incluso en señales cortas ajustando n_win dinámicamente.
    """
    x = signal.samples
    n = len(x)
    if n == 0 or signal.fs <= 0:
        raise ValueError("Señal vacía o frecuencia de muestreo no positiva.")

    n_win = min(max(16, int(window_size)), n)
    fft_size = next_power_of_two(n_win)
    step = max(1, min(int(hop_size), n_win))

    weights, coherent_gain = build_window(n_win, window)
    norm = n_win * coherent_gain

    num_frames = max(1, 1 + (n - n_win) // step)
    num_bins = fft_size // 2 + 1
    mag_matrix = np.zeros((num_frames, num_bins), dtype=np.float64)
    times = np.zeros(num_frames, dtype=np.float64)

    for idx in range(num_frames):
        start = idx * step
        segment = x[start : start + n_win]
        if len(segment) < n_win:
            padded = np.zeros(n_win, dtype=np.float64)
            padded[: len(segment)] = segment
            segment = padded
        frame_win = segment * weights
        spec = np.abs(np.fft.rfft(frame_win, n=fft_size)) / norm
        if len(spec) > 2:
            spec[1:-1] *= 2.0
        mag_matrix[idx, :] = spec
        times[idx] = (start + n_win / 2.0) / signal.fs

    peak = max(float(np.max(mag_matrix)), 1e-12)
    power_db = 20.0 * np.log10(np.clip(mag_matrix / peak, 1e-5, 1.0))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / signal.fs)

    return {
        "times": times,
        "frequencies": freqs,
        "power_db": power_db,
        "num_frames": num_frames,
        "num_bins": num_bins,
    }
