"""
Módulo de Codificación de Línea en Banda Base (Baseband Line Coding) — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C — RF-08)

Implementa la conversión de secuencias binarias (bits PCM o comprimidos)
a formas de onda discretas x[n] aproximando pulsos continuos en banda base:
- Unipolar NRZ (Non-Return-to-Zero Unipolar)
- Polar NRZ-L (Non-Return-to-Zero Polar Level)
- Manchester (Bifase con transición garantizada a mitad de bit Tb/2)

Incluye cálculo de reloj de símbolo Tb y reporte de ancho de banda analítico (primer nulo).
Cumple con el contrato inmutable ModulatedSignal congelado en dsp_engine/contracts.py.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import LineCodingType, ModulatedSignal


def encode_baseband(
    bits: Union[List[int], NDArray[np.uint8], NDArray[np.int32], NDArray[np.int64]],
    line_coding: LineCodingType = "polar-nrz",
    symbol_rate_baud: float = 1000.0,
    samples_per_symbol: int = 16,
    amplitude: float = 1.0,
) -> ModulatedSignal:
    """
    Codifica una secuencia binaria en una forma de onda de pulsos en banda base.

    Parámetros:
    -----------
    bits : Secuencia unidimensional de bits (0 y 1).
    line_coding : Tipo de codificación ('unipolar-nrz', 'polar-nrz', 'manchester').
    symbol_rate_baud : Tasa de símbolos/bits Rb en Baudios (bits/s). Rb > 0.
    samples_per_symbol : Número de muestras discretas por símbolo (N_sps >= 2).
                         Para Manchester debe ser un número par.
    amplitude : Amplitud pico de los pulsos en voltios (A > 0).

    Retorna:
    --------
    ModulatedSignal con:
      - scheme = 'BB'
      - carrier_hz = 0.0 (Banda Base sin modulación sobre portadora)
      - fs = symbol_rate_baud * samples_per_symbol
      - estimated_bandwidth_hz = Rb para NRZ (primer nulo), 2*Rb para Manchester
      - ideal_constellation con los niveles I/Q discretos de decisión
    """
    bit_arr = np.asarray(bits, dtype=np.uint8).ravel()

    if bit_arr.size == 0:
        raise ValueError("La secuencia de bits no puede estar vacía.")

    if not np.all((bit_arr == 0) | (bit_arr == 1)):
        raise ValueError("La secuencia de entrada solo debe contener valores binarios 0 o 1.")

    if symbol_rate_baud <= 0.0:
        raise ValueError(f"La tasa de símbolos debe ser estrictamente positiva: {symbol_rate_baud} Baud.")

    if samples_per_symbol < 2:
        raise ValueError(f"samples_per_symbol debe ser >= 2, recibido: {samples_per_symbol}.")

    if amplitude <= 0.0:
        raise ValueError(f"La amplitud debe ser positiva: {amplitude}.")

    amp = float(amplitude)
    sps = int(samples_per_symbol)
    rb = float(symbol_rate_baud)
    fs = rb * sps

    if line_coding == "unipolar-nrz":
        # Bit 1 -> +A, Bit 0 -> 0.0 V
        symbol_levels = np.where(bit_arr == 1, amp, 0.0).astype(np.float64)
        waveform = np.repeat(symbol_levels, sps)
        estimated_bw = rb
        ideal_constellation: Optional[List[Tuple[float, float]]] = [(0.0, 0.0), (amp, 0.0)]

    elif line_coding == "polar-nrz":
        # Bit 1 -> +A, Bit 0 -> -A V (NRZ-L)
        symbol_levels = np.where(bit_arr == 1, amp, -amp).astype(np.float64)
        waveform = np.repeat(symbol_levels, sps)
        estimated_bw = rb
        ideal_constellation = [(-amp, 0.0), (amp, 0.0)]

    elif line_coding == "manchester":
        # Manchester Bifase:
        # Bit 1: Primer semi-intervalo +A, Segundo semi-intervalo -A
        # Bit 0: Primer semi-intervalo -A, Segundo semi-intervalo +A
        # Garantiza transición en Tb/2 y componente DC promedio nula.
        if sps % 2 != 0:
            raise ValueError(
                f"Para codificación Manchester, samples_per_symbol debe ser un número par (recibido {sps})."
            )
        half_sps = sps // 2
        first_half = np.where(bit_arr == 1, amp, -amp).astype(np.float64)
        second_half = np.where(bit_arr == 1, -amp, amp).astype(np.float64)

        # Intercalar semi-pulsos y repetir por half_sps
        interleaved = np.column_stack([first_half, second_half]).ravel()
        waveform = np.repeat(interleaved, half_sps)
        estimated_bw = 2.0 * rb
        ideal_constellation = [(-amp, 0.0), (amp, 0.0)]

    else:
        raise ValueError(f"Esquema de codificación de línea no soportado: '{line_coding}'.")

    return ModulatedSignal(
        scheme="BB",
        waveform=waveform.astype(np.float64),
        fs=float(fs),
        carrier_hz=0.0,
        symbol_rate_baud=rb,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=float(estimated_bw),
        line_coding=line_coding,
        modulation_index=None,
        ideal_constellation=ideal_constellation,
    )


def generate_symbol_clock(
    num_symbols: int,
    samples_per_symbol: int,
    high_level: float = 1.0,
    low_level: float = 0.0,
) -> NDArray[np.float64]:
    """
    Genera la señal de reloj cuadrada síncrona con el tiempo de bit Tb.

    Parámetros:
    -----------
    num_symbols : Número de símbolos (bits) transmitidos.
    samples_per_symbol : Muestras por símbolo N_sps (debe ser >= 2).
    high_level : Amplitud del semi-ciclo alto (por defecto 1.0 V).
    low_level : Amplitud del semi-ciclo bajo (por defecto 0.0 V).

    Retorna:
    --------
    Arreglo NumPy 1D de tamaño num_symbols * samples_per_symbol con ciclo de trabajo al 50%.
    """
    if num_symbols <= 0:
        raise ValueError("num_symbols debe ser >= 1.")
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol debe ser >= 2.")

    half_sps = samples_per_symbol // 2
    rem_sps = samples_per_symbol - half_sps

    single_clock_cycle = np.concatenate([
        np.full(half_sps, high_level, dtype=np.float64),
        np.full(rem_sps, low_level, dtype=np.float64),
    ])

    return np.tile(single_clock_cycle, num_symbols)
