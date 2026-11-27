"""Demoduladores analógicos AM y FM que entregan contratos inmutables."""

import math
from typing import Dict, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from dsp_engine.contracts import DemodulationOutput, ModulatedSignal
from dsp_engine.demodulation.filters import lowpass_filter

AnalogMethod = Literal["hilbert", "rectifier"]
_ANALOG_SCHEMES = ("AM-LC", "AM-SC", "FM")


def _validate_modulated(modulated: ModulatedSignal) -> tuple[NDArray[np.float64], float]:
    """Normaliza la forma de onda y valida la frecuencia de muestreo."""
    fs = float(modulated.fs)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"Frecuencia de muestreo inválida: {modulated.fs} Hz.")
    samples = np.asarray(modulated.waveform, dtype=np.float64).reshape(-1)
    return samples, fs


def _validate_filter_order(order: int) -> int:
    """Valida el orden Butterworth solicitado por el receptor."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"Orden de filtro inválido: {order}. Debe ser entero >= 1.")
    return order


def _empty_output() -> DemodulationOutput:
    """Construye una salida analógica vacía con tipos definidos."""
    return DemodulationOutput(
        recovered_bits=np.empty(0, dtype=np.uint8),
        recovered_signal=np.empty(0, dtype=np.float64),
        rx_constellation=None,
        eye_diagram_traces=None,
    )


def _analog_output(values: NDArray[np.float64]) -> DemodulationOutput:
    """Empaqueta la señal recuperada sin bits ni constelación."""
    return DemodulationOutput(
        recovered_bits=np.empty(0, dtype=np.uint8),
        recovered_signal=np.asarray(values, dtype=np.float64).reshape(-1),
        rx_constellation=None,
        eye_diagram_traces=None,
    )


def _remove_dc(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Resta la media para eliminar la componente continua."""
    if values.size == 0:
        return values.copy()
    return np.asarray(values - float(np.mean(values)), dtype=np.float64)


def _default_cutoff(modulated: ModulatedSignal, fs: float, mode: str) -> float:
    """Estima el ancho de banda del mensaje a partir del contrato de modulación."""
    bandwidth = float(modulated.estimated_bandwidth_hz)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        rate = float(modulated.symbol_rate_baud)
        bandwidth = rate if math.isfinite(rate) and rate > 0.0 else 0.2 * fs
    cutoff = 0.5 * bandwidth
    if mode == "am":
        carrier = abs(float(modulated.carrier_hz))
        if math.isfinite(carrier) and carrier > 0.0:
            cutoff = min(cutoff, 0.45 * carrier)
    cutoff = min(cutoff, 0.45 * fs)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        cutoff = 0.1 * fs
    return cutoff


def _resolve_cutoff(modulated: ModulatedSignal, cutoff_hz: Optional[float], fs: float, mode: str) -> float:
    """Usa el corte explícito o el estimado, siempre por debajo de Nyquist."""
    if cutoff_hz is not None:
        cutoff = float(cutoff_hz)
        if not math.isfinite(cutoff) or not 0.0 < cutoff < fs / 2.0:
            raise ValueError(f"Corte inválido: {cutoff_hz} Hz. Debe estar en (0, Fs/2).")
        return cutoff
    return _default_cutoff(modulated, fs, mode)


def _apply_lpf(values: NDArray[np.float64], cutoff: float, fs: float, order: int) -> NDArray[np.float64]:
    """Aplica el pasabajos de fase cero solo cuando hay muestras suficientes."""
    if values.size < 2:
        return np.asarray(values, dtype=np.float64).copy()
    return lowpass_filter(values, cutoff, fs, order)


def _scale_by_mod_index(values: NDArray[np.float64], modulated: ModulatedSignal) -> NDArray[np.float64]:
    """Deshace el índice de modulación AM cuando está disponible y es positivo."""
    index = modulated.modulation_index
    if index is None:
        return values
    index_value = float(index)
    if math.isfinite(index_value) and index_value > 0.0:
        return np.asarray(values / index_value, dtype=np.float64)
    return values


def demodulate_am_envelope(
    modulated: ModulatedSignal,
    cutoff_hz: Optional[float] = None,
    method: AnalogMethod = "hilbert",
    filter_order: int = 4,
) -> DemodulationOutput:
    """Recupera la envolvente AM: |hilbert(x)| o |x| seguido de LPF y eliminación de DC."""
    samples, fs = _validate_modulated(modulated)
    if samples.size == 0:
        return _empty_output()

    method_value = str(method)
    if method_value not in ("hilbert", "rectifier"):
        raise ValueError(f"Método de envolvente no soportado: {method}")
    order = _validate_filter_order(filter_order)

    if method_value == "hilbert":
        envelope = np.abs(signal.hilbert(samples)).astype(np.float64)
        recovered = _remove_dc(envelope)
        if cutoff_hz is not None:
            cutoff = _resolve_cutoff(modulated, cutoff_hz, fs, "am")
            recovered = _apply_lpf(recovered, cutoff, fs, order)
    else:
        cutoff = _resolve_cutoff(modulated, cutoff_hz, fs, "am")
        recovered = _apply_lpf(np.abs(samples), cutoff, fs, order)
        recovered = _remove_dc(recovered)

    return _analog_output(_scale_by_mod_index(recovered, modulated))


def demodulate_am_coherent(
    modulated: ModulatedSignal,
    cutoff_hz: Optional[float] = None,
    phase_rad: float = 0.0,
    filter_order: int = 4,
) -> DemodulationOutput:
    """Mezcla con 2cos(2*pi*fc*t + phi) y filtra el término en 2*fc."""
    samples, fs = _validate_modulated(modulated)
    if samples.size == 0:
        return _empty_output()

    carrier = float(modulated.carrier_hz)
    phase = float(phase_rad)
    if not math.isfinite(carrier) or carrier <= 0.0:
        raise ValueError(f"Portadora inválida para AM coherente: {modulated.carrier_hz} Hz.")
    if not math.isfinite(phase):
        raise ValueError("La fase local debe ser finita.")
    order = _validate_filter_order(filter_order)
    cutoff = _resolve_cutoff(modulated, cutoff_hz, fs, "am")

    t = np.arange(samples.size, dtype=np.float64) / fs
    local_carrier = 2.0 * np.cos(2.0 * np.pi * carrier * t + phase)
    recovered = lowpass_filter(samples * local_carrier, cutoff, fs, order)
    if modulated.scheme == "AM-LC":
        recovered = _remove_dc(recovered)
    return _analog_output(recovered)


def demodulate_fm(
    modulated: ModulatedSignal,
    freq_deviation_hz: Optional[float] = None,
    cutoff_hz: Optional[float] = None,
    filter_order: int = 4,
) -> DemodulationOutput:
    """Discrimina f_i = (Fs/2pi)*d/dt arg(z[n]) - fc con z[n] = hilbert(x[n])."""
    samples, fs = _validate_modulated(modulated)
    if samples.size == 0:
        return _empty_output()

    carrier = float(modulated.carrier_hz)
    if not math.isfinite(carrier) or carrier < 0.0:
        raise ValueError(f"Portadora inválida para FM: {modulated.carrier_hz} Hz.")
    order = _validate_filter_order(filter_order)
    deviation: Optional[float] = None
    if freq_deviation_hz is not None:
        deviation = float(freq_deviation_hz)
        if not math.isfinite(deviation) or deviation <= 0.0:
            raise ValueError(f"Desviación de frecuencia inválida: {freq_deviation_hz} Hz.")

    analytic = signal.hilbert(samples)
    phase = np.unwrap(np.angle(analytic)).astype(np.float64)
    instantaneous = (fs / (2.0 * np.pi)) * np.diff(phase) - carrier
    if instantaneous.size == 0:
        instantaneous = np.zeros(samples.size, dtype=np.float64)
    else:
        instantaneous = np.append(instantaneous, instantaneous[-1]).astype(np.float64)

    if deviation is None:
        low, high = np.percentile(instantaneous, [2.0, 98.0])
        estimate = abs(float(high) - float(low)) / 2.0
        deviation = estimate if estimate > 1e-12 else None
    normalized = instantaneous if deviation is None else instantaneous / deviation

    cutoff = _resolve_cutoff(modulated, cutoff_hz, fs, "fm")
    recovered = _apply_lpf(normalized, cutoff, fs, order)
    return _analog_output(recovered)


def _optional_float_kwarg(kwargs: Dict[str, object], name: str) -> Optional[float]:
    """Lee un argumento opcional convertido a float."""
    value = kwargs.get(name)
    return None if value is None else float(value)


def _float_kwarg(kwargs: Dict[str, object], name: str, default: float) -> float:
    """Lee un argumento float con valor por defecto."""
    value = kwargs.get(name, default)
    return default if value is None else float(value)


def _int_kwarg(kwargs: Dict[str, object], name: str, default: int) -> int:
    """Lee un argumento entero con valor por defecto."""
    value = kwargs.get(name, default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} debe ser entero.")
    return int(value)


def demodulate_analog(modulated: ModulatedSignal, **kwargs: object) -> DemodulationOutput:
    """Despacha AM-LC, AM-SC y FM al demodulador analógico correspondiente."""
    scheme = modulated.scheme
    if scheme not in _ANALOG_SCHEMES:
        raise ValueError(f"Esquema analógico no soportado: {scheme}")

    cutoff = _optional_float_kwarg(kwargs, "cutoff_hz")
    order = _int_kwarg(kwargs, "filter_order", 4)
    if scheme == "FM":
        return demodulate_fm(
            modulated,
            freq_deviation_hz=_optional_float_kwarg(kwargs, "freq_deviation_hz"),
            cutoff_hz=cutoff,
            filter_order=order,
        )
    if scheme == "AM-SC":
        return demodulate_am_coherent(
            modulated,
            cutoff_hz=cutoff,
            phase_rad=_float_kwarg(kwargs, "phase_rad", 0.0),
            filter_order=order,
        )

    method_value = str(kwargs.get("method", "hilbert"))
    if method_value not in ("hilbert", "rectifier"):
        raise ValueError(f"Método de envolvente no soportado: {method_value}")
    method = cast(AnalogMethod, method_value)
    return demodulate_am_envelope(modulated, cutoff_hz=cutoff, method=method, filter_order=order)
