"""Muestreo temporal x[n] y diagnóstico del Teorema de Nyquist-Shannon (NQS)."""

from typing import Optional
import numpy as np

from dsp_engine.contracts import SamplingStudyResult, Signal1D
from dsp_engine.core.fft import compute_alias_frequency, compute_spectrum


def simulate_sampling_nqs(
    signal: Signal1D,
    downsample_factor: int = 1,
    known_max_freq_hz: Optional[float] = None,
) -> SamplingStudyResult:
    """
    Simula el muestreo discreto x[n] por diezmado directo sin filtro antialiasing previo
    para demostrar cuantitativamente el criterio de Nyquist-Shannon y el aliasing.
    """
    factor = max(1, int(downsample_factor))
    original_fs = float(signal.fs)
    effective_fs = original_fs / factor
    effective_ts = 1.0 / effective_fs

    indices = np.arange(0, len(signal.samples), factor)
    sampled_values = signal.samples[indices].astype(np.float64)
    sampled_times = (indices / original_fs).astype(np.float64)

    f_max_signal = (
        float(known_max_freq_hz)
        if known_max_freq_hz is not None
        else compute_spectrum(signal, window="hann", energy_threshold=0.99).f_max_hz
    )
    nyquist_min = 2.0 * f_max_signal
    nyquist_ok = bool(effective_fs >= nyquist_min)
    nyquist_margin = effective_fs - nyquist_min
    alias_freq = None if nyquist_ok else compute_alias_frequency(f_max_signal, effective_fs)

    return SamplingStudyResult(
        original_fs=original_fs,
        effective_fs=effective_fs,
        effective_ts=effective_ts,
        downsample_factor=factor,
        sampled_values=sampled_values,
        sampled_times=sampled_times,
        f_max_signal_hz=f_max_signal,
        nyquist_minimum_hz=nyquist_min,
        nyquist_satisfied=nyquist_ok,
        nyquist_margin_hz=nyquist_margin,
        alias_frequency_hz=alias_freq,
    )
