"""Análisis espectral FFT unilateral, ventaneo coherente y cálculo exacto de aliasing."""

from typing import Optional, Tuple
import numpy as np

from dsp_engine.contracts import Signal1D, SpectralAnalysis, WindowType


def next_power_of_two(n: int) -> int:
    """Retorna la menor potencia de 2 mayor o igual a n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def build_window(n: int, window_type: WindowType) -> Tuple[np.ndarray, float]:
    """Genera coeficientes de ventana espectral y su ganancia coherente."""
    if n <= 0:
        raise ValueError("La longitud de la ventana debe ser positiva.")
    if n == 1:
        return np.ones(1, dtype=np.float64), 1.0

    if window_type == "hann":
        weights = np.hanning(n).astype(np.float64)
    elif window_type == "hamming":
        weights = np.hamming(n).astype(np.float64)
    elif window_type == "blackman":
        weights = np.blackman(n).astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    coherent_gain = float(np.mean(weights))
    return weights, (coherent_gain if coherent_gain > 0 else 1.0)


def compute_alias_frequency(signal_freq_hz: float, fs: float) -> float:
    """Calcula la frecuencia aparente reflejada por aliasing: f_alias = |f0 - round(f0/Fs)*Fs|."""
    if fs <= 0:
        return 0.0
    k = round(signal_freq_hz / fs)
    return float(abs(signal_freq_hz - k * fs))


def compute_spectrum(
    signal: Signal1D,
    window: WindowType = "hann",
    energy_threshold: float = 0.99,
    reference_max_freq_hz: Optional[float] = None,
) -> SpectralAnalysis:
    """
    Calcula la Transformada Rápida de Fourier (FFT) unilateral calibrada en amplitud,
    estimando f_max al porcentaje de energía especificado, criterio NQS y BW efectivo.
    """
    x = signal.samples
    n = len(x)
    fs = signal.fs
    if n == 0 or fs <= 0:
        raise ValueError("Señal vacía o frecuencia de muestreo no positiva.")

    weights, coherent_gain = build_window(n, window)
    x_win = x * weights

    fft_size = next_power_of_two(n)
    fft_vals = np.fft.rfft(x_win, n=fft_size)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / fs).astype(np.float64)

    norm_factor = n * coherent_gain
    magnitudes = (np.abs(fft_vals) / norm_factor).astype(np.float64)
    if len(magnitudes) > 2:
        magnitudes[1:-1] *= 2.0

    phases = np.angle(fft_vals).astype(np.float64)

    # Frecuencia dominante AC (ignorando DC en k=0 si existen componentes AC)
    if len(magnitudes) > 1:
        dominant_idx = int(np.argmax(magnitudes[1:])) + 1
    else:
        dominant_idx = 0

    peak_ref = max(float(np.max(magnitudes)), 1e-12)
    magnitudes_db = (20.0 * np.log10(np.clip(magnitudes / peak_ref, 1e-6, 1.0))).astype(
        np.float64
    )

    energy = magnitudes ** 2
    total_energy = float(np.sum(energy))
    if total_energy <= 1e-15:
        f_max_measured = float(frequencies[-1])
    else:
        cum_energy = np.cumsum(energy) / total_energy
        f_max_idx = min(int(np.searchsorted(cum_energy, energy_threshold)), len(frequencies) - 1)
        f_max_measured = float(frequencies[f_max_idx])

    eff_ref_max = reference_max_freq_hz if reference_max_freq_hz is not None else f_max_measured
    nyquist_rate_hz = 2.0 * eff_ref_max
    is_aliasing = bool(fs < nyquist_rate_hz)
    alias_freq = (
        compute_alias_frequency(reference_max_freq_hz, fs)
        if is_aliasing and reference_max_freq_hz is not None
        else None
    )

    return SpectralAnalysis(
        fft_size=fft_size,
        window=window,
        frequencies=frequencies,
        magnitudes=magnitudes,
        magnitudes_db=magnitudes_db,
        phases=phases,
        total_energy=total_energy,
        f_max_hz=f_max_measured,
        nyquist_rate_hz=nyquist_rate_hz,
        is_aliasing=is_aliasing,
        alias_frequency_hz=alias_freq,
        bandwidth_hz=f_max_measured,
        dominant_frequency_hz=float(frequencies[dominant_idx]),
    )
