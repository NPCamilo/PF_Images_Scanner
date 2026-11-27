"""Métricas de auditoría TX/RX con manejo finito y seguro de casos borde."""

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import AuditMetrics

_CAP_DB = 120.0
_EPS = 1e-12


def _aligned_signals(
    tx_signal: NDArray[np.float64],
    rx_signal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Alinea ambas señales al mínimo común de muestras para evitar desbordes."""
    tx = np.asarray(tx_signal, dtype=np.float64).reshape(-1)
    rx = np.asarray(rx_signal, dtype=np.float64).reshape(-1)
    length = min(tx.size, rx.size)
    return tx[:length], rx[:length]


def compute_mse(tx_signal: NDArray[np.float64], rx_signal: NDArray[np.float64]) -> float:
    """Calcula MSE = media((tx - rx)^2)."""
    tx, rx = _aligned_signals(tx_signal, rx_signal)
    if tx.size == 0:
        return 0.0
    error = tx - rx
    return float(np.mean(error * error))


def compute_rmse(tx_signal: NDArray[np.float64], rx_signal: NDArray[np.float64]) -> float:
    """Calcula RMSE = sqrt(MSE)."""
    return math.sqrt(compute_mse(tx_signal, rx_signal))


def compute_snr_db(tx_signal: NDArray[np.float64], rx_signal: NDArray[np.float64]) -> float:
    """Calcula SNR = 10log10(P_tx/P_error) con saturación finita en dB."""
    tx, rx = _aligned_signals(tx_signal, rx_signal)
    if tx.size == 0:
        return 0.0
    error_power = compute_mse(tx, rx)
    signal_power = float(np.mean(tx * tx))
    if signal_power <= _EPS and error_power <= _EPS:
        return _CAP_DB
    if signal_power <= _EPS:
        return -_CAP_DB
    if error_power <= _EPS:
        return _CAP_DB
    return float(np.clip(10.0 * math.log10(signal_power / error_power), -_CAP_DB, _CAP_DB))


def compute_psnr_db(
    tx_signal: NDArray[np.float64],
    rx_signal: NDArray[np.float64],
    peak_value: float = 1.0,
) -> float:
    """Calcula PSNR = 10log10(peak^2/MSE) con saturación finita en dB."""
    peak = float(peak_value)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"peak_value inválido: {peak_value}.")
    error_power = compute_mse(tx_signal, rx_signal)
    if error_power <= _EPS:
        return _CAP_DB
    value = 10.0 * math.log10((peak * peak) / error_power)
    return float(np.clip(value, -_CAP_DB, _CAP_DB))


def compute_ber(tx_bits: NDArray[np.uint8], rx_bits: NDArray[np.uint8]) -> float:
    """Calcula BER = errores / bits comparados sobre la longitud común."""
    tx = np.asarray(tx_bits, dtype=np.uint8).reshape(-1) & 1
    rx = np.asarray(rx_bits, dtype=np.uint8).reshape(-1) & 1
    length = min(tx.size, rx.size)
    if length == 0:
        return 0.0
    return float(np.count_nonzero(tx[:length] != rx[:length]) / length)


def compute_pearson_correlation(
    tx_signal: NDArray[np.float64],
    rx_signal: NDArray[np.float64],
) -> float:
    """Calcula rho = cov(tx,rx)/(std(tx)*std(rx)), con regla para señales constantes."""
    tx, rx = _aligned_signals(tx_signal, rx_signal)
    if tx.size == 0:
        return 0.0
    if tx.size == 1:
        return 1.0 if tx[0] == rx[0] else 0.0

    tx_constant = bool(np.all(tx == tx[0]))
    rx_constant = bool(np.all(rx == rx[0]))
    if tx_constant or rx_constant:
        return 1.0 if tx_constant and rx_constant and tx[0] == rx[0] else 0.0

    correlation = float(np.corrcoef(tx, rx)[0, 1])
    if not math.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, -1.0, 1.0))


def compute_audit_metrics(
    tx_signal: Optional[NDArray[np.float64]] = None,
    rx_signal: Optional[NDArray[np.float64]] = None,
    tx_bits: Optional[NDArray[np.uint8]] = None,
    rx_bits: Optional[NDArray[np.uint8]] = None,
    peak_value: float = 1.0,
) -> AuditMetrics:
    """Integra fidelidad de señal y de bits en la estructura AuditMetrics."""
    if tx_signal is None or rx_signal is None:
        mse = 0.0
        rmse = 0.0
        snr_db = 0.0
        psnr_db = 0.0
        correlation = 0.0
    else:
        mse = compute_mse(tx_signal, rx_signal)
        rmse = math.sqrt(mse)
        snr_db = compute_snr_db(tx_signal, rx_signal)
        psnr_db = compute_psnr_db(tx_signal, rx_signal, peak_value)
        correlation = compute_pearson_correlation(tx_signal, rx_signal)

    if tx_bits is None or rx_bits is None:
        bit_errors = 0
        total_bits = 0
        ber = 0.0
    else:
        tx = np.asarray(tx_bits, dtype=np.uint8).reshape(-1) & 1
        rx = np.asarray(rx_bits, dtype=np.uint8).reshape(-1) & 1
        total_bits = int(min(tx.size, rx.size))
        bit_errors = int(np.count_nonzero(tx[:total_bits] != rx[:total_bits]))
        ber = float(bit_errors / total_bits) if total_bits > 0 else 0.0

    return AuditMetrics(
        mse=float(mse),
        rmse=float(rmse),
        snr_db=float(snr_db),
        psnr_db=float(psnr_db),
        ber=ber,
        bit_errors=bit_errors,
        total_bits=total_bits,
        pearson_correlation=float(correlation),
    )
