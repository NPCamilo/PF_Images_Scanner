"""Demoduladores digitales con correlación por símbolo y trazas de ojo serializables."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import DemodulationOutput, ModulatedSignal
from dsp_engine.demodulation.analog_demod import demodulate_analog
from dsp_engine.demodulation.filters import lowpass_filter

_DIGITAL_SCHEMES = ("BB", "ASK", "FSK", "BPSK", "QPSK")
_GRAY_BITS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1), (1, 0))


def _validate_positive_int(value: int, name: str) -> int:
    """Valida un entero positivo usado como longitud o orden."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} debe ser entero >= 1, recibido: {value}.")
    return value


def _validate_signal(modulated: ModulatedSignal) -> tuple[NDArray[np.float64], float, int, int]:
    """Normaliza la señal y deriva la cantidad de símbolos completos."""
    fs = float(modulated.fs)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"Frecuencia de muestreo inválida: {modulated.fs} Hz.")
    samples = np.asarray(modulated.waveform, dtype=np.float64).reshape(-1)
    sps = _validate_positive_int(int(modulated.samples_per_symbol), "samples_per_symbol")
    num_symbols = samples.size // sps
    return samples, fs, sps, num_symbols


def _empty_output() -> DemodulationOutput:
    """Construye una salida digital vacía con arreglos y listas tipados."""
    return DemodulationOutput(
        recovered_bits=np.empty(0, dtype=np.uint8),
        recovered_signal=np.empty(0, dtype=np.float64),
        rx_constellation=[],
        eye_diagram_traces=[],
    )


def _make_output(
    bits: NDArray[np.uint8],
    recovered: NDArray[np.float64],
    constellation: List[Tuple[float, float]],
    traces: List[List[float]],
) -> DemodulationOutput:
    """Empaqueta la decisión digital en la estructura inmutable del proyecto."""
    return DemodulationOutput(
        recovered_bits=np.asarray(bits, dtype=np.uint8).reshape(-1),
        recovered_signal=np.asarray(recovered, dtype=np.float64).reshape(-1),
        rx_constellation=constellation,
        eye_diagram_traces=traces,
    )


def extract_eye_diagram_traces(
    signal: NDArray[np.float64],
    samples_per_symbol: int,
    max_traces: int = 64,
) -> List[List[float]]:
    """Toma ventanas de 2*Ts con paso Ts y las convierte a listas de floats para JSON."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    sps = _validate_positive_int(samples_per_symbol, "samples_per_symbol")
    if isinstance(max_traces, bool) or not isinstance(max_traces, int) or max_traces < 0:
        raise ValueError(f"max_traces inválido: {max_traces}.")
    if max_traces == 0:
        return []

    width = 2 * sps
    starts = np.arange(0, max(0, x.size - width + 1), sps, dtype=np.int64)
    if starts.size == 0:
        return []
    if starts.size > max_traces:
        chosen = np.unique(np.linspace(0, starts.size - 1, max_traces, dtype=np.int64))
        starts = starts[chosen]
    return [x[start : start + width].astype(np.float64).tolist() for start in starts]


def _symbol_matrix(samples: NDArray[np.float64], sps: int, num_symbols: int) -> NDArray[np.float64]:
    """Reagrupa las muestras en filas de un símbolo, descartando el resto final."""
    if num_symbols == 0:
        return np.empty((0, sps), dtype=np.float64)
    return np.asarray(samples[: num_symbols * sps].reshape(num_symbols, sps), dtype=np.float64)


def _adaptive_threshold(values: NDArray[np.float64]) -> float:
    """Calcula el punto medio entre niveles extremos, o 0.5 si son idénticos."""
    if values.size == 0:
        return 0.5
    low = float(np.min(values))
    high = float(np.max(values))
    return (low + high) / 2.0 if high > low else 0.5


def _carrier_correlation(
    samples: NDArray[np.float64],
    sps: int,
    num_symbols: int,
    fs: float,
    carrier_hz: float,
    phase_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Correlación I/Q por símbolo: I = mean(2x cos), Q = mean(-2x sin)."""
    if num_symbols == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy()
    t = np.arange(sps, dtype=np.float64) / fs
    angle = 2.0 * np.pi * carrier_hz * t + phase_rad
    cos_ref = (2.0 * np.cos(angle)).astype(np.float64)
    sin_ref = (-2.0 * np.sin(angle)).astype(np.float64)
    matrix = _symbol_matrix(samples, sps, num_symbols)
    i_values = np.asarray(matrix @ cos_ref / sps, dtype=np.float64)
    q_values = np.asarray(matrix @ sin_ref / sps, dtype=np.float64)
    return i_values, q_values


def _filtered_baseband(
    modulated: ModulatedSignal,
    samples: NDArray[np.float64],
    fs: float,
    phase_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mezcla I/Q y elimina el término en 2*fc con un Butterworth de fase cero."""
    carrier = float(modulated.carrier_hz)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise ValueError(f"Portadora inválida: {modulated.carrier_hz} Hz.")
    if not math.isfinite(phase_rad):
        raise ValueError("La fase local debe ser finita.")

    rate = float(modulated.symbol_rate_baud)
    bandwidth = float(modulated.estimated_bandwidth_hz)
    if math.isfinite(bandwidth) and bandwidth > 0.0:
        cutoff = bandwidth
    elif math.isfinite(rate) and rate > 0.0:
        cutoff = rate
    else:
        cutoff = 0.25 * fs
    cutoff = min(cutoff, 0.45 * fs)

    t = np.arange(samples.size, dtype=np.float64) / fs
    angle = 2.0 * np.pi * carrier * t + phase_rad
    i_wave = lowpass_filter(2.0 * samples * np.cos(angle), cutoff, fs, 4)
    q_wave = lowpass_filter(-2.0 * samples * np.sin(angle), cutoff, fs, 4)
    return i_wave, q_wave


def demodulate_baseband(modulated: ModulatedSignal) -> DemodulationOutput:
    """Integra cada símbolo y decide con umbral adaptativo según la codificación de línea."""
    samples, fs, sps, num_symbols = _validate_signal(modulated)
    if num_symbols == 0:
        return _empty_output()

    coding = modulated.line_coding if modulated.line_coding is not None else "polar-nrz"
    if coding not in ("unipolar-nrz", "polar-nrz", "manchester"):
        raise ValueError(f"Codificación de línea no soportada: {coding}")
    if coding == "manchester" and sps < 2:
        raise ValueError("Manchester requiere al menos 2 muestras por símbolo.")

    matrix = _symbol_matrix(samples, sps, num_symbols)
    if coding == "manchester":
        half = sps // 2
        first_half = np.asarray(np.mean(matrix[:, :half], axis=1), dtype=np.float64)
        second_half = np.asarray(np.mean(matrix[:, half:], axis=1), dtype=np.float64)
        values = first_half - second_half
    else:
        values = np.asarray(np.mean(matrix, axis=1), dtype=np.float64)

    threshold = _adaptive_threshold(values)
    bits = (values >= threshold).astype(np.uint8)
    recovered = np.asarray(np.repeat(values, sps)[: samples.size], dtype=np.float64)
    constellation = [(float(value), 0.0) for value in values]
    traces = extract_eye_diagram_traces(recovered, sps)
    return _make_output(bits, recovered, constellation, traces)


def demodulate_ask(modulated: ModulatedSignal, threshold: Optional[float] = None) -> DemodulationOutput:
    """Detecta la envolvente por símbolo con I/Q y aplica un umbral adaptativo."""
    samples, fs, sps, num_symbols = _validate_signal(modulated)
    if num_symbols == 0:
        return _empty_output()

    carrier = float(modulated.carrier_hz)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise ValueError(f"Portadora inválida para ASK: {modulated.carrier_hz} Hz.")

    i_values, q_values = _carrier_correlation(samples, sps, num_symbols, fs, carrier, 0.0)
    envelope = np.asarray(np.hypot(i_values, q_values), dtype=np.float64)
    decision_level = _adaptive_threshold(envelope) if threshold is None else float(threshold)
    if not math.isfinite(decision_level):
        raise ValueError("El umbral de ASK debe ser finito.")
    bits = (envelope >= decision_level).astype(np.uint8)

    i_wave, q_wave = _filtered_baseband(modulated, samples, fs, 0.0)
    recovered = np.asarray(np.hypot(i_wave, q_wave), dtype=np.float64)
    constellation = [(float(value), 0.0) for value in i_values]
    traces = extract_eye_diagram_traces(recovered, sps)
    return _make_output(bits, recovered, constellation, traces)


def _resolve_fsk_frequencies(
    modulated: ModulatedSignal,
    fs: float,
    sps: int,
    f0_hz: Optional[float],
    f1_hz: Optional[float],
) -> tuple[float, float]:
    """Por defecto usa f0 = fc - Rs/2 y f1 = fc + Rs/2, con Rs = Fs/sps."""
    rate = float(modulated.symbol_rate_baud)
    if not math.isfinite(rate) or rate <= 0.0:
        rate = fs / sps
    carrier = float(modulated.carrier_hz)
    has_carrier = math.isfinite(carrier) and carrier > 0.0

    if f0_hz is None and f1_hz is None:
        if not has_carrier:
            raise ValueError("Se requiere carrier_hz para inferir las frecuencias FSK.")
        low = carrier - rate / 2.0
        high = carrier + rate / 2.0
    elif f0_hz is None:
        high = float(f1_hz)
        low = 2.0 * carrier - high if has_carrier else high - rate
    elif f1_hz is None:
        low = float(f0_hz)
        high = 2.0 * carrier - low if has_carrier else low + rate
    else:
        low = float(f0_hz)
        high = float(f1_hz)

    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Las frecuencias FSK deben ser finitas.")
    if not 0.0 < low < fs / 2.0 or not 0.0 < high < fs / 2.0 or low == high:
        raise ValueError(f"Frecuencias FSK inválidas: {low}, {high} Hz con Fs={fs} Hz.")
    return low, high


def demodulate_fsk(
    modulated: ModulatedSignal,
    f0_hz: Optional[float] = None,
    f1_hz: Optional[float] = None,
) -> DemodulationOutput:
    """Compara energías de correladores en cuadratura: E1 > E0 implica bit 1."""
    samples, fs, sps, num_symbols = _validate_signal(modulated)
    if num_symbols == 0:
        return _empty_output()

    low, high = _resolve_fsk_frequencies(modulated, fs, sps, f0_hz, f1_hz)
    matrix = _symbol_matrix(samples, sps, num_symbols)
    t = np.arange(sps, dtype=np.float64) / fs
    corr0 = matrix @ np.exp(-2j * np.pi * low * t)
    corr1 = matrix @ np.exp(-2j * np.pi * high * t)
    energy0 = np.asarray(np.abs(corr0) ** 2, dtype=np.float64)
    energy1 = np.asarray(np.abs(corr1) ** 2, dtype=np.float64)

    metric = energy1 - energy0
    bits = (energy1 > energy0).astype(np.uint8)
    recovered = np.asarray(np.repeat(metric, sps)[: samples.size], dtype=np.float64)
    constellation = [(float(value), 0.0) for value in metric]
    traces = extract_eye_diagram_traces(recovered, sps)
    return _make_output(bits, recovered, constellation, traces)


def demodulate_bpsk(modulated: ModulatedSignal, phase_rad: float = 0.0) -> DemodulationOutput:
    """Decide I >= 0 como bit 1 tras correlacionar con cos y -sin de la portadora."""
    samples, fs, sps, num_symbols = _validate_signal(modulated)
    if num_symbols == 0:
        return _empty_output()

    carrier = float(modulated.carrier_hz)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise ValueError(f"Portadora inválida para BPSK: {modulated.carrier_hz} Hz.")
    if not math.isfinite(phase_rad):
        raise ValueError("La fase local debe ser finita.")

    i_values, q_values = _carrier_correlation(samples, sps, num_symbols, fs, carrier, phase_rad)
    bits = (i_values >= 0.0).astype(np.uint8)
    i_wave, _ = _filtered_baseband(modulated, samples, fs, phase_rad)
    constellation = [(float(i), float(q)) for i, q in zip(i_values, q_values)]
    traces = extract_eye_diagram_traces(i_wave, sps)
    return _make_output(bits, i_wave, constellation, traces)


def demodulate_qpsk(modulated: ModulatedSignal, phase_rad: float = 0.0) -> DemodulationOutput:
    """Decodifica 2 bits por símbolo con Gray: (+,+)=00, (-,+)=01, (-,-)=11, (+,-)=10."""
    samples, fs, sps, num_symbols = _validate_signal(modulated)
    if num_symbols == 0:
        return _empty_output()

    carrier = float(modulated.carrier_hz)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise ValueError(f"Portadora inválida para QPSK: {modulated.carrier_hz} Hz.")
    if not math.isfinite(phase_rad):
        raise ValueError("La fase local debe ser finita.")

    i_values, q_values = _carrier_correlation(samples, sps, num_symbols, fs, carrier, phase_rad)
    ideal = modulated.ideal_constellation
    if ideal is None:
        i_positive = i_values >= 0.0
        q_positive = q_values >= 0.0
        indices = np.where(
            i_positive & q_positive,
            0,
            np.where(~i_positive & q_positive, 1, np.where(~i_positive & ~q_positive, 2, 3)),
        )
    else:
        points = np.asarray(ideal, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] == 0:
            raise ValueError("ideal_constellation no puede estar vacía.")
        vectors = np.column_stack((i_values, q_values))
        distances = np.sum((vectors[:, None, :] - points[None, :, :]) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)

    pairs = np.asarray([_GRAY_BITS[int(index) % 4] for index in indices], dtype=np.uint8)
    bits = pairs.reshape(-1)
    i_wave, _ = _filtered_baseband(modulated, samples, fs, phase_rad)
    constellation = [(float(i), float(q)) for i, q in zip(i_values, q_values)]
    traces = extract_eye_diagram_traces(i_wave, sps)
    return _make_output(bits, i_wave, constellation, traces)


def _optional_float_kwarg(kwargs: Dict[str, object], name: str) -> Optional[float]:
    """Lee un argumento opcional convertido a float."""
    value = kwargs.get(name)
    return None if value is None else float(value)


def _float_kwarg(kwargs: Dict[str, object], name: str, default: float) -> float:
    """Lee un argumento float con valor por defecto."""
    value = kwargs.get(name, default)
    return default if value is None else float(value)


def demodulate_digital(modulated: ModulatedSignal, **kwargs: object) -> DemodulationOutput:
    """Despacha BB, ASK, FSK, BPSK y QPSK según el esquema del contrato."""
    scheme = modulated.scheme
    if scheme not in _DIGITAL_SCHEMES:
        raise ValueError(f"Esquema digital no soportado: {scheme}")
    if scheme == "BB":
        return demodulate_baseband(modulated)
    if scheme == "ASK":
        return demodulate_ask(modulated, threshold=_optional_float_kwarg(kwargs, "threshold"))
    if scheme == "FSK":
        return demodulate_fsk(
            modulated,
            f0_hz=_optional_float_kwarg(kwargs, "f0_hz"),
            f1_hz=_optional_float_kwarg(kwargs, "f1_hz"),
        )
    if scheme == "BPSK":
        return demodulate_bpsk(modulated, phase_rad=_float_kwarg(kwargs, "phase_rad", 0.0))
    return demodulate_qpsk(modulated, phase_rad=_float_kwarg(kwargs, "phase_rad", 0.0))


def demodulate_signal(modulated: ModulatedSignal, **kwargs: object) -> DemodulationOutput:
    """Punto de entrada maestro: elige el receptor analógico o digital según scheme."""
    scheme = modulated.scheme
    if scheme in ("AM-LC", "AM-SC", "FM"):
        return demodulate_analog(modulated, **kwargs)
    if scheme in _DIGITAL_SCHEMES:
        return demodulate_digital(modulated, **kwargs)
    raise ValueError(f"Esquema de modulación no soportado: {scheme}")
