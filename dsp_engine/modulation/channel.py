"""
Módulo de Simulación de Canal AWGN — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C — RF-10)

Implementa:
- Generador de Ruido Blanco Gaussiano Aditivo (AWGN) mediante Transformada de Box-Muller.
- Calibración exacta de relación señal-a-ruido de canal (SNR_canal en dB) sobre la potencia real de la señal.
- Soporte para modo ideal sin ruido (SNR = inf).
- Filtro pasabanda / pasobajo opcional para simular limitación de ancho de banda del canal físico.
- Retorno conforme al contrato inmutable ChannelOutput congelado en dsp_engine/contracts.py.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from dsp_engine.contracts import ChannelOutput, ModulatedSignal, Signal1D


def generate_box_muller_noise(
    num_samples: int,
    std_dev: float = 1.0,
    mean: float = 0.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Genera muestras de ruido gaussiano blanco utilizando la Transformada de Box-Muller:
      Z0 = sqrt(-2 * ln(U1)) * cos(2*pi * U2)
      Z1 = sqrt(-2 * ln(U1)) * sin(2*pi * U2)
      con U1, U2 ~ Uniforme(0, 1) independientes.

    Parámetros:
    -----------
    num_samples : Cantidad de muestras a generar (N >= 1).
    std_dev : Desviación estándar sigma del ruido (sigma >= 0).
    mean : Media mu del ruido (por defecto 0.0 V).
    seed : Semilla pseudoaleatoria para pruebas determinísticas reproducibles.

    Retorna:
    --------
    Arreglo NumPy 1D de tipo float64 con distribución N(mean, std_dev^2).
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples debe ser >= 1 (recibido {num_samples}).")
    if std_dev < 0.0:
        raise ValueError(f"std_dev debe ser >= 0.0 (recibido {std_dev}).")
    if std_dev == 0.0:
        return np.full(num_samples, mean, dtype=np.float64)

    rng = np.random.default_rng(seed)
    half_count = (num_samples + 1) // 2

    # U1 en intervalo (1e-12, 1.0] para evitar ln(0)
    u1 = rng.uniform(1e-12, 1.0, half_count)
    u2 = rng.uniform(0.0, 1.0, half_count)

    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2

    z0 = radius * np.cos(angle)
    z1 = radius * np.sin(angle)

    interleaved = np.column_stack([z0, z1]).ravel()[:num_samples]
    return (mean + std_dev * interleaved).astype(np.float64)


def _apply_channel_bandwidth_filter(
    waveform: NDArray[np.float64],
    fs: float,
    filter_bandwidth_hz: float,
    center_freq_hz: float = 0.0,
    order: int = 4,
) -> NDArray[np.float64]:
    """Filtro digital Butterworth (SOS con fase cero filtfilt) limitador de ancho de banda del canal."""
    nyquist = fs / 2.0
    if nyquist <= 0.0:
        return waveform

    bw = float(filter_bandwidth_hz)
    fc = float(center_freq_hz)

    if fc > 0.0:
        low = max(0.01 * nyquist, fc - bw / 2.0)
        high = min(0.99 * nyquist, fc + bw / 2.0)
        if low >= high:
            return waveform
        sos = sp_signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    else:
        cutoff = min(0.99 * nyquist, max(0.01 * nyquist, bw))
        sos = sp_signal.butter(order, cutoff / nyquist, btype="lowpass", output="sos")

    return sp_signal.sosfiltfilt(sos, waveform).astype(np.float64)


def simulate_awgn_channel(
    signal: Union[ModulatedSignal, Signal1D, NDArray[np.float64], list],
    target_snr_db: float,
    seed: Optional[int] = None,
    filter_bandwidth_hz: Optional[float] = None,
    center_freq_hz: Optional[float] = None,
    fs: Optional[float] = None,
) -> ChannelOutput:
    """
    Simula el paso de una señal por un Canal de Ruido Blanco Gaussiano Aditivo (AWGN).

    Parámetros:
    -----------
    signal : Señal transmitida (ModulatedSignal, Signal1D o arreglo de muestras).
    target_snr_db : SNR objetivo en decibeles [0..40 dB].
                    Si target_snr_db = inf o math.isinf(...), se activa el modo ideal sin ruido.
    seed : Semilla pseudoaleatoria para reproducibilidad.
    filter_bandwidth_hz : Ancho de banda de corte del canal (opcional para simular limitación de canal).
    center_freq_hz : Frecuencia central del canal (por defecto heredada de ModulatedSignal o 0.0).
    fs : Frecuencia de muestreo (por defecto heredada de ModulatedSignal/Signal1D).

    Retorna:
    --------
    ChannelOutput con:
      - noisy_waveform: Señal recibida s[n] + w[n] (opcionalmente filtrada por el canal)
      - noise_added: Componente de ruido exacto w[n] = noisy_waveform - signal
      - target_snr_db: SNR configurada
      - actual_snr_db: SNR medida experimentalmente: 10 * log10(P_signal / P_ruido_real)
    """
    # Extracción polimórfica de muestras y parámetros
    if isinstance(signal, ModulatedSignal):
        raw_waveform = np.asarray(signal.waveform, dtype=np.float64)
        channel_fs = float(signal.fs)
        cf = float(signal.carrier_hz) if center_freq_hz is None else float(center_freq_hz)
        fbw = float(signal.estimated_bandwidth_hz) if filter_bandwidth_hz is None else float(filter_bandwidth_hz)
    elif isinstance(signal, Signal1D):
        raw_waveform = np.asarray(signal.samples, dtype=np.float64)
        channel_fs = float(signal.fs)
        cf = 0.0 if center_freq_hz is None else float(center_freq_hz)
        fbw = filter_bandwidth_hz
    else:
        raw_waveform = np.asarray(signal, dtype=np.float64).ravel()
        channel_fs = float(fs) if fs is not None else 8000.0
        cf = 0.0 if center_freq_hz is None else float(center_freq_hz)
        fbw = filter_bandwidth_hz

    if raw_waveform.size == 0:
        raise ValueError("La señal transmitida no puede estar vacía.")

    num_samples = raw_waveform.size
    signal_power = float(np.mean(raw_waveform ** 2))

    # Caso modo ideal: SNR = +inf
    if math.isinf(target_snr_db) and target_snr_db > 0.0:
        zero_noise = np.zeros(num_samples, dtype=np.float64)
        return ChannelOutput(
            noisy_waveform=raw_waveform.copy(),
            noise_added=zero_noise,
            target_snr_db=float("inf"),
            actual_snr_db=float("inf"),
        )

    if target_snr_db < -30.0:
        raise ValueError(f"SNR objetivo no válida: {target_snr_db} dB. Debe ser >= -30 dB o inf.")

    if signal_power <= 1e-15:
        raise ValueError(
            "La potencia media de la señal transmitida es prácticamente nula. "
            "No es posible calibrar la SNR sobre una señal plana en cero."
        )

    # Cálculo analítico de potencia y desviación estándar del ruido:
    # SNR_lineal = P_sig / P_noise = 10^(SNR_db / 10) -> P_noise = P_sig / 10^(SNR_db / 10)
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    sigma_noise = math.sqrt(target_noise_power)

    # Generación de ruido gaussiano vía Box-Muller
    noise_samples = generate_box_muller_noise(
        num_samples=num_samples,
        std_dev=sigma_noise,
        mean=0.0,
        seed=seed,
    )

    noisy_waveform = raw_waveform + noise_samples

    # Aplicación opcional de filtro limitador de ancho de banda de canal
    if fbw is not None and fbw > 0.0:
        filtered_noisy = _apply_channel_bandwidth_filter(
            waveform=noisy_waveform,
            fs=channel_fs,
            filter_bandwidth_hz=fbw,
            center_freq_hz=cf,
        )
        effective_noise = filtered_noisy - raw_waveform
        noisy_waveform = filtered_noisy
    else:
        effective_noise = noise_samples

    # Medición cuantitativa del SNR real inyectado
    actual_noise_power = float(np.mean(effective_noise ** 2))
    if actual_noise_power <= 1e-15:
        actual_snr_db = float("inf")
    else:
        actual_snr_db = float(10.0 * np.log10(signal_power / actual_noise_power))

    return ChannelOutput(
        noisy_waveform=noisy_waveform.astype(np.float64),
        noise_added=effective_noise.astype(np.float64),
        target_snr_db=float(target_snr_db),
        actual_snr_db=float(actual_snr_db),
    )
