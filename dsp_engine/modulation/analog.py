"""
Módulo de Modulación Continua / Portadora CW Analógica — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C — RF-09)

Implementa los moduladores analógicos sobre portadora continua:
- AM DSB-LC (Doble Banda Lateral con Gran Portadora / Large Carrier)
- AM DSB-SC (Doble Banda Lateral con Portadora Suprimida / Suppressed Carrier)
- FM (Modulación de Frecuencia con cálculo exacto de BW por Regla de Carson)

Garantiza traslación espectral hacia +-fc y cumplimiento del contrato ModulatedSignal.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import ModulatedSignal, Signal1D


def compute_carson_bandwidth(frequency_deviation_hz: float, message_f_max_hz: float) -> float:
    """
    Calcula el Ancho de Banda de transmisión de una señal FM según la Regla de Carson:
    BW ≈ 2 * (Δf + f_m) = 2 * f_m * (β + 1)
    """
    if frequency_deviation_hz <= 0.0:
        raise ValueError(f"La desviación de frecuencia Δf debe ser > 0: {frequency_deviation_hz} Hz.")
    if message_f_max_hz <= 0.0:
        raise ValueError(f"La frecuencia máxima del mensaje f_m debe ser > 0: {message_f_max_hz} Hz.")

    return float(2.0 * (frequency_deviation_hz + message_f_max_hz))


def _extract_samples_and_fs(
    signal: Union[Signal1D, NDArray[np.float64], list],
    fs: Optional[float] = None,
    message_f_max_hz: Optional[float] = None,
) -> Tuple[NDArray[np.float64], float, float]:
    """Auxiliar interno para desempaquetar muestras, fs y f_max de forma unificada."""
    if isinstance(signal, Signal1D):
        samples = np.asarray(signal.samples, dtype=np.float64)
        sample_rate = float(signal.fs)
        if message_f_max_hz is not None:
            fm = float(message_f_max_hz)
        elif "frequency_hz" in signal.metadata:
            fm = float(signal.metadata["frequency_hz"])
        else:
            # Estimación preliminar: frecuencia Nyquist / 8 o espectro
            fm = max(1.0, sample_rate / 16.0)
    else:
        samples = np.asarray(signal, dtype=np.float64).ravel()
        if fs is None or fs <= 0.0:
            raise ValueError("Debe especificarse una frecuencia de muestreo fs > 0 cuando se pasa un arreglo.")
        sample_rate = float(fs)
        fm = float(message_f_max_hz) if message_f_max_hz is not None and message_f_max_hz > 0.0 else max(1.0, sample_rate / 16.0)

    if samples.size == 0:
        raise ValueError("La señal moduladora no puede estar vacía.")

    if sample_rate <= 0.0:
        raise ValueError(f"Frecuencia de muestreo inválida: {sample_rate} Hz.")

    return samples, sample_rate, fm


def modulate_am_lc(
    signal: Union[Signal1D, NDArray[np.float64], list],
    carrier_hz: float,
    modulation_index: float = 0.8,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
    fs: Optional[float] = None,
    message_f_max_hz: Optional[float] = None,
) -> ModulatedSignal:
    """
    Modulación de Amplitud con Gran Portadora (AM DSB-LC).

    Ecuación:
      s(t) = A_c * [1 + m * m_n(t)] * cos(2*pi*f_c*t + phi_0)
      donde m_n(t) es el mensaje normalizado a pico unitario: m(t) / max(|m(t)|).

    Ancho de Banda de transmisión:
      BW = 2 * f_m
    """
    samples, sample_rate, fm = _extract_samples_and_fs(signal, fs, message_f_max_hz)

    if carrier_hz <= 0.0:
        raise ValueError(f"La frecuencia de portadora fc debe ser > 0: {carrier_hz} Hz.")
    if carrier_amplitude <= 0.0:
        raise ValueError(f"La amplitud de portadora Ac debe ser > 0: {carrier_amplitude} V.")
    if modulation_index < 0.0:
        raise ValueError(f"El índice de modulación m debe ser >= 0: {modulation_index}.")

    bw = 2.0 * fm
    highest_freq = carrier_hz + bw / 2.0
    if sample_rate < 2.0 * highest_freq:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente: fs={sample_rate} Hz < 2*(fc + BW/2)={2.0 * highest_freq} Hz. "
            f"Aumente fs para evitar aliasing en la portadora o reduzca fc."
        )

    # Normalización del mensaje modulador m_n(t) con pico unitario
    peak = float(np.max(np.abs(samples)))
    m_n = samples / peak if peak > 1e-12 else np.zeros_like(samples)

    n = samples.size
    t = np.arange(n, dtype=np.float64) / sample_rate
    carrier = np.cos(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)
    envelope = carrier_amplitude * (1.0 + modulation_index * m_n)
    waveform = envelope * carrier

    sps = max(1, int(round(sample_rate / fm)))

    return ModulatedSignal(
        scheme="AM-LC",
        waveform=waveform.astype(np.float64),
        fs=sample_rate,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=fm,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(bw),
        line_coding=None,
        modulation_index=float(modulation_index),
        ideal_constellation=None,
    )


def modulate_am_sc(
    signal: Union[Signal1D, NDArray[np.float64], list],
    carrier_hz: float,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
    fs: Optional[float] = None,
    message_f_max_hz: Optional[float] = None,
) -> ModulatedSignal:
    """
    Modulación de Amplitud con Portadora Suprimida (AM DSB-SC).

    Ecuación:
      s(t) = A_c * m_n(t) * cos(2*pi*f_c*t + phi_0)
      donde la portadora discreta en f_c queda completamente suprimida.

    Ancho de Banda de transmisión:
      BW = 2 * f_m
    """
    samples, sample_rate, fm = _extract_samples_and_fs(signal, fs, message_f_max_hz)

    if carrier_hz <= 0.0:
        raise ValueError(f"La frecuencia de portadora fc debe ser > 0: {carrier_hz} Hz.")
    if carrier_amplitude <= 0.0:
        raise ValueError(f"La amplitud de portadora Ac debe ser > 0: {carrier_amplitude} V.")

    bw = 2.0 * fm
    highest_freq = carrier_hz + bw / 2.0
    if sample_rate < 2.0 * highest_freq:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente: fs={sample_rate} Hz < 2*(fc + BW/2)={2.0 * highest_freq} Hz. "
            f"Aumente fs para evitar aliasing en la portadora o reduzca fc."
        )

    peak = float(np.max(np.abs(samples)))
    m_n = samples / peak if peak > 1e-12 else np.zeros_like(samples)

    n = samples.size
    t = np.arange(n, dtype=np.float64) / sample_rate
    carrier = np.cos(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)
    waveform = carrier_amplitude * m_n * carrier

    sps = max(1, int(round(sample_rate / fm)))

    return ModulatedSignal(
        scheme="AM-SC",
        waveform=waveform.astype(np.float64),
        fs=sample_rate,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=fm,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(bw),
        line_coding=None,
        modulation_index=None,
        ideal_constellation=None,
    )


def modulate_fm(
    signal: Union[Signal1D, NDArray[np.float64], list],
    carrier_hz: float,
    frequency_deviation_hz: float = 75.0,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
    fs: Optional[float] = None,
    message_f_max_hz: Optional[float] = None,
) -> ModulatedSignal:
    """
    Modulación de Frecuencia (FM) con cálculo exacto de ancho de banda por Regla de Carson.

    Ecuación en tiempo continuo:
      s(t) = A_c * cos(2*pi*f_c*t + 2*pi*Δf * integral_0^t m_n(tau) dtau + phi_0)

    Aproximación discreta:
      integral_0^t m_n(tau) dtau ≈ T_s * sum_{k=0}^n m_n[k]

    Índice de modulación:
      β = Δf / f_m

    Ancho de Banda de Carson:
      BW_Carson = 2 * (Δf + f_m) = 2 * f_m * (β + 1)
    """
    samples, sample_rate, fm = _extract_samples_and_fs(signal, fs, message_f_max_hz)

    if carrier_hz <= 0.0:
        raise ValueError(f"La frecuencia de portadora fc debe ser > 0: {carrier_hz} Hz.")
    if carrier_amplitude <= 0.0:
        raise ValueError(f"La amplitud de portadora Ac debe ser > 0: {carrier_amplitude} V.")
    if frequency_deviation_hz <= 0.0:
        raise ValueError(f"La desviación de frecuencia Δf debe ser > 0: {frequency_deviation_hz} Hz.")

    bw_carson = compute_carson_bandwidth(frequency_deviation_hz, fm)
    beta = frequency_deviation_hz / fm

    highest_freq = carrier_hz + bw_carson / 2.0
    if sample_rate < 2.0 * highest_freq:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente para FM: fs={sample_rate} Hz < 2*(fc + BW_Carson/2)={2.0 * highest_freq} Hz. "
            f"Aumente fs para cumplir el límite de Carson sin aliasing."
        )

    peak = float(np.max(np.abs(samples)))
    m_n = samples / peak if peak > 1e-12 else np.zeros_like(samples)

    ts = 1.0 / sample_rate
    n = samples.size
    t = np.arange(n, dtype=np.float64) * ts

    # Integración numérica acumulada con paso rectangular
    phase_integral = np.cumsum(m_n) * ts
    instantaneous_phase = 2.0 * np.pi * carrier_hz * t + 2.0 * np.pi * frequency_deviation_hz * phase_integral + carrier_phase_rad
    waveform = carrier_amplitude * np.cos(instantaneous_phase)

    sps = max(1, int(round(sample_rate / fm)))

    return ModulatedSignal(
        scheme="FM",
        waveform=waveform.astype(np.float64),
        fs=sample_rate,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=fm,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(bw_carson),
        line_coding=None,
        modulation_index=float(beta),
        ideal_constellation=None,
    )
