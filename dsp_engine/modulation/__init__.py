"""
Módulo de Modulación Banda Base, Portadora CW y Canal AWGN — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C)

Exportaciones principales:
- Codificación en Banda Base: encode_baseband, generate_symbol_clock
- Modulación CW Analógica: modulate_am_lc, modulate_am_sc, modulate_fm, compute_carson_bandwidth
- Modulación Digital Pasobanda: modulate_ask, modulate_fsk, modulate_bpsk, modulate_qpsk
"""

from .baseband import encode_baseband, generate_symbol_clock
from .analog import (
    compute_carson_bandwidth,
    modulate_am_lc,
    modulate_am_sc,
    modulate_fm,
)
from .digital import (
    modulate_ask,
    modulate_bpsk,
    modulate_fsk,
    modulate_qpsk,
)

MODULATION_MODULE_OWNER = "Ángel Gabriel Burbano"

__all__ = [
    "MODULATION_MODULE_OWNER",
    "encode_baseband",
    "generate_symbol_clock",
    "modulate_am_lc",
    "modulate_am_sc",
    "modulate_fm",
    "compute_carson_bandwidth",
    "modulate_ask",
    "modulate_bpsk",
    "modulate_fsk",
    "modulate_qpsk",
]
