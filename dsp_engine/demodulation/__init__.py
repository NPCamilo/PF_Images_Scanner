"""
Módulo de Demodulación, Filtrado Butterworth, Reconstrucción DAC y Auditoría
Responsable asignado: Luis Ospina (feat/demodulation-luis)

Contratos en dsp_engine/contracts.py:
- DemodulationOutput, AuditMetrics, Signal1D
"""

from .filters import (
    apply_zero_phase_filter,
    bandpass_filter,
    design_butterworth_bpf,
    design_butterworth_lpf,
    lowpass_filter,
)
from .analog_demod import (
    demodulate_am_coherent,
    demodulate_am_envelope,
    demodulate_analog,
    demodulate_fm,
)
from .digital_demod import (
    demodulate_ask,
    demodulate_baseband,
    demodulate_bpsk,
    demodulate_digital,
    demodulate_fsk,
    demodulate_qpsk,
    demodulate_signal,
    extract_eye_diagram_traces,
)
from .reconstruction import reconstruct_dac, reconstruct_sinc, reconstruct_zoh
from .metrics import (
    compute_audit_metrics,
    compute_ber,
    compute_mse,
    compute_pearson_correlation,
    compute_psnr_db,
    compute_rmse,
    compute_snr_db,
)

DEMODULATION_MODULE_OWNER = "Luis Ospina"

__all__ = [
    "design_butterworth_lpf",
    "design_butterworth_bpf",
    "apply_zero_phase_filter",
    "lowpass_filter",
    "bandpass_filter",
    "demodulate_am_envelope",
    "demodulate_am_coherent",
    "demodulate_fm",
    "demodulate_analog",
    "extract_eye_diagram_traces",
    "demodulate_baseband",
    "demodulate_ask",
    "demodulate_fsk",
    "demodulate_bpsk",
    "demodulate_qpsk",
    "demodulate_digital",
    "demodulate_signal",
    "reconstruct_sinc",
    "reconstruct_zoh",
    "reconstruct_dac",
    "compute_mse",
    "compute_rmse",
    "compute_snr_db",
    "compute_psnr_db",
    "compute_ber",
    "compute_pearson_correlation",
    "compute_audit_metrics",
]
