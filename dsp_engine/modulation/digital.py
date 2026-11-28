"""
Módulo de Modulaciones Digitales Pasobanda y Constelaciones I/Q — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C — RF-09)

Implementa los moduladores digitales pasobanda:
- ASK (Amplitude Shift Keying / OOK)
- BFSK (Binary Frequency Shift Keying en modo audible para Modo 1 y pasabanda)
- BPSK (Binary Phase Shift Keying)
- QPSK (Quadrature Phase Shift Keying / 4-QAM con codificación Gray)

Genera mapeos analíticos de constelación ideal (I, Q) y cumple con el contrato ModulatedSignal.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import ModulatedSignal


def _validate_digital_inputs(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    carrier_hz: float,
    symbol_rate_baud: float,
    samples_per_symbol: int,
    carrier_amplitude: float,
    estimated_bw_hz: float,
) -> Tuple[NDArray[np.uint8], float, int, float, float]:
    """Valida entradas comunes de moduladores digitales y verifica el límite de Nyquist."""
    bit_arr = np.asarray(bits, dtype=np.uint8).ravel()

    if bit_arr.size == 0:
        raise ValueError("La secuencia de bits no puede estar vacía.")

    if not np.all((bit_arr == 0) | (bit_arr == 1)):
        raise ValueError("La secuencia solo debe contener valores binarios 0 o 1.")

    if carrier_hz <= 0.0:
        raise ValueError(f"La frecuencia de portadora fc debe ser > 0: {carrier_hz} Hz.")

    if symbol_rate_baud <= 0.0:
        raise ValueError(f"La tasa de símbolos debe ser > 0: {symbol_rate_baud} Baud.")

    if samples_per_symbol < 2:
        raise ValueError(f"samples_per_symbol debe ser >= 2 (recibido {samples_per_symbol}).")

    if carrier_amplitude <= 0.0:
        raise ValueError(f"La amplitud de portadora debe ser > 0: {carrier_amplitude} V.")

    fs = float(symbol_rate_baud * samples_per_symbol)
    highest_freq = carrier_hz + estimated_bw_hz / 2.0

    if fs < 2.0 * highest_freq:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente: fs={fs} Hz < 2*(fc + BW/2)={2.0 * highest_freq} Hz. "
            f"Aumente samples_per_symbol ({samples_per_symbol}) para evitar aliasing en la portadora digital."
        )

    return bit_arr, float(symbol_rate_baud), int(samples_per_symbol), float(carrier_amplitude), fs


def modulate_ask(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    carrier_hz: float,
    symbol_rate_baud: float = 1000.0,
    samples_per_symbol: int = 16,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
) -> ModulatedSignal:
    """
    Modulación por Desplazamiento de Amplitud Binaria (2-ASK / OOK - On-Off Keying).

    Ecuación:
      s(t) = A_c * b[k] * cos(2*pi*f_c*t + phi_0),  con b[k] ∈ {0, 1}

    Ancho de Banda (primer nulo):
      BW = 2 * R_b
    Constelación Ideal:
      [(0.0, 0.0), (A_c, 0.0)]
    """
    estimated_bw = 2.0 * symbol_rate_baud
    bit_arr, rb, sps, ac, fs = _validate_digital_inputs(
        bits, carrier_hz, symbol_rate_baud, samples_per_symbol, carrier_amplitude, estimated_bw
    )

    n_samples = bit_arr.size * sps
    t = np.arange(n_samples, dtype=np.float64) / fs
    carrier = np.cos(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)
    baseband_levels = np.repeat(bit_arr.astype(np.float64), sps)
    waveform = ac * baseband_levels * carrier

    ideal_constellation: List[Tuple[float, float]] = [(0.0, 0.0), (ac, 0.0)]

    return ModulatedSignal(
        scheme="ASK",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=rb,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(estimated_bw),
        line_coding=None,
        modulation_index=None,
        ideal_constellation=ideal_constellation,
    )


def modulate_bpsk(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    carrier_hz: float,
    symbol_rate_baud: float = 1000.0,
    samples_per_symbol: int = 16,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
) -> ModulatedSignal:
    """
    Modulación por Desplazamiento de Fase Binaria (BPSK).

    Ecuación:
      s(t) = A_c * cos(2*pi*f_c*t + theta_k + phi_0)
      donde theta_k = 0 para bit 1, y theta_k = pi para bit 0
      Equivalente en banda base polar: s(t) = A_c * d[k] * cos(2*pi*f_c*t + phi_0), d[k] ∈ {-1, +1}

    Ancho de Banda (primer nulo):
      BW = 2 * R_b
    Constelación Ideal:
      [(-A_c, 0.0), (A_c, 0.0)]
    """
    estimated_bw = 2.0 * symbol_rate_baud
    bit_arr, rb, sps, ac, fs = _validate_digital_inputs(
        bits, carrier_hz, symbol_rate_baud, samples_per_symbol, carrier_amplitude, estimated_bw
    )

    n_samples = bit_arr.size * sps
    t = np.arange(n_samples, dtype=np.float64) / fs
    carrier = np.cos(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)
    # Mapeo: bit 1 -> +1, bit 0 -> -1
    polar_levels = np.repeat(np.where(bit_arr == 1, 1.0, -1.0), sps)
    waveform = ac * polar_levels * carrier

    ideal_constellation: List[Tuple[float, float]] = [(-ac, 0.0), (ac, 0.0)]

    return ModulatedSignal(
        scheme="BPSK",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=rb,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(estimated_bw),
        line_coding=None,
        modulation_index=None,
        ideal_constellation=ideal_constellation,
    )


def modulate_fsk(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    carrier_hz: Optional[float] = None,
    f0_hz: Optional[float] = None,
    f1_hz: Optional[float] = None,
    freq_shift_hz: Optional[float] = None,
    symbol_rate_baud: float = 300.0,
    samples_per_symbol: int = 32,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
    continuous_phase: bool = True,
) -> ModulatedSignal:
    """
    Modulación por Desplazamiento de Frecuencia Binario (BFSK).

    Soporta:
      - Modo Audible (Modo 1 de Nexus-DSP para escuchar datos por altavoz): f0=1200 Hz, f1=2200 Hz (estándar Bell 202).
      - Modo Pasabanda de Portadora Alta: fc +/- delta_f.
      - Fase Continua (CPFSK): Elimina discontinuidades abruptas en los límites de símbolo.

    Ancho de Banda estimado:
      BW ≈ |f1 - f0| + 2 * R_b
    """
    bit_arr = np.asarray(bits, dtype=np.uint8).ravel()
    if bit_arr.size == 0:
        raise ValueError("La secuencia de bits no puede estar vacía.")
    if not np.all((bit_arr == 0) | (bit_arr == 1)):
        raise ValueError("La secuencia solo debe contener valores binarios 0 o 1.")
    if symbol_rate_baud <= 0.0:
        raise ValueError(f"symbol_rate_baud debe ser > 0: {symbol_rate_baud}")
    if samples_per_symbol < 2:
        raise ValueError(f"samples_per_symbol debe ser >= 2: {samples_per_symbol}")
    if carrier_amplitude <= 0.0:
        raise ValueError(f"carrier_amplitude debe ser > 0: {carrier_amplitude}")

    # Resolución de frecuencias f0 y f1
    if f0_hz is not None and f1_hz is not None:
        f0 = float(f0_hz)
        f1 = float(f1_hz)
        fc = (f0 + f1) / 2.0
    elif carrier_hz is not None:
        fc = float(carrier_hz)
        shift = float(freq_shift_hz) if freq_shift_hz is not None else float(symbol_rate_baud)
        f0 = fc - shift
        f1 = fc + shift
    else:
        # Frecuencias audibles por defecto (Bell 202 AFSK)
        f0 = 1200.0
        f1 = 2200.0
        fc = 1700.0

    if f0 <= 0.0 or f1 <= 0.0:
        raise ValueError(f"Las frecuencias FSK f0={f0} Hz y f1={f1} Hz deben ser estrictamente positivas.")

    estimated_bw = abs(f1 - f0) + 2.0 * symbol_rate_baud
    fs = float(symbol_rate_baud * samples_per_symbol)
    highest_freq = max(f0, f1) + symbol_rate_baud

    if fs < 2.0 * highest_freq:
        raise ValueError(
            f"Frecuencia de muestreo insuficiente para FSK: fs={fs} Hz < 2*f_max={2.0 * highest_freq} Hz. "
            f"Aumente samples_per_symbol ({samples_per_symbol}) para simular los tonos FSK sin aliasing."
        )

    sps = int(samples_per_symbol)
    ac = float(carrier_amplitude)
    rb = float(symbol_rate_baud)

    freq_per_bit = np.where(bit_arr == 1, f1, f0).astype(np.float64)
    instant_freqs = np.repeat(freq_per_bit, sps)

    ts = 1.0 / fs
    n_samples = instant_freqs.size

    if continuous_phase:
        # CPFSK: Integración numérica acumulada de la frecuencia instantánea
        phase = 2.0 * np.pi * np.cumsum(instant_freqs) * ts + carrier_phase_rad
        waveform = ac * np.cos(phase)
    else:
        # FSK de fase no continua
        t = np.arange(n_samples, dtype=np.float64) * ts
        waveform = ac * np.cos(2.0 * np.pi * instant_freqs * t + carrier_phase_rad)

    ideal_constellation: List[Tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]

    return ModulatedSignal(
        scheme="FSK",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=float(fc),
        symbol_rate_baud=rb,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(estimated_bw),
        line_coding=None,
        modulation_index=abs(f1 - f0) / rb,
        ideal_constellation=ideal_constellation,
    )


def modulate_qpsk(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    carrier_hz: float,
    symbol_rate_baud: float = 1000.0,
    samples_per_symbol: int = 16,
    carrier_amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
) -> ModulatedSignal:
    """
    Modulación por Desplazamiento de Fase en Cuadratura (QPSK / 4-QAM).

    Mapeo Gray ortogonal I/Q:
      Dibit (b0, b1):
        b0 -> Rama en Fase (I): +d si b0=1, -d si b0=0
        b1 -> Rama en Cuadratura (Q): +d si b1=1, -d si b1=0
      donde d = A_c / sqrt(2) para mantener la envolvente constante: sqrt(I^2 + Q^2) = A_c.

    Forma de onda en el aire:
      s(t) = I(t) * cos(2*pi*f_c*t + phi_0) - Q(t) * sin(2*pi*f_c*t + phi_0)

    Tasa de símbolos:
      R_s = R_b / 2
    Ancho de Banda (primer nulo):
      BW = 2 * R_s = R_b (la mitad del ancho de banda de BPSK para la misma tasa binaria)
    Constelación Ideal:
      [(-d, -d), (-d, d), (d, -d), (d, d)]
    """
    bit_arr = np.asarray(bits, dtype=np.uint8).ravel()

    if bit_arr.size == 0:
        raise ValueError("La secuencia de bits no puede estar vacía.")

    if not np.all((bit_arr == 0) | (bit_arr == 1)):
        raise ValueError("La secuencia solo debe contener valores binarios 0 o 1.")

    # Relleno con 0 si la longitud de bits es impar para completar los dibits
    if bit_arr.size % 2 != 0:
        bit_arr = np.append(bit_arr, 0)

    estimated_bw = 2.0 * (symbol_rate_baud / 2.0)  # BW = Rb
    _, rs, sps, ac, fs = _validate_digital_inputs(
        bit_arr, carrier_hz, symbol_rate_baud / 2.0, samples_per_symbol, carrier_amplitude, estimated_bw
    )

    b_i = bit_arr[0::2]
    b_q = bit_arr[1::2]
    num_symbols = b_i.size

    d = ac / np.sqrt(2.0)
    i_symbols = np.where(b_i == 1, d, -d).astype(np.float64)
    q_symbols = np.where(b_q == 1, d, -d).astype(np.float64)

    i_waveform = np.repeat(i_symbols, sps)
    q_waveform = np.repeat(q_symbols, sps)

    n_samples = num_symbols * sps
    t = np.arange(n_samples, dtype=np.float64) / fs

    carrier_i = np.cos(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)
    carrier_q = np.sin(2.0 * np.pi * carrier_hz * t + carrier_phase_rad)

    # s(t) = I * cos(wc*t) - Q * sin(wc*t)
    waveform = i_waveform * carrier_i - q_waveform * carrier_q

    ideal_constellation: List[Tuple[float, float]] = [
        (-d, -d),
        (-d, d),
        (d, -d),
        (d, d),
    ]

    return ModulatedSignal(
        scheme="QPSK",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=float(carrier_hz),
        symbol_rate_baud=rs,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(estimated_bw),
        line_coding=None,
        modulation_index=None,
        ideal_constellation=ideal_constellation,
    )
