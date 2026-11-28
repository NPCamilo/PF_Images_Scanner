"""
Módulo de Modulación Banda Base, Portadora CW y Canal AWGN — Nexus-DSP
Responsable: Ángel Gabriel Burbano (Bloque C)

Exportaciones principales:
- Codificación en Banda Base: encode_baseband, generate_symbol_clock
"""

from .baseband import encode_baseband, generate_symbol_clock

MODULATION_MODULE_OWNER = "Ángel Gabriel Burbano"

__all__ = [
    "MODULATION_MODULE_OWNER",
    "encode_baseband",
    "generate_symbol_clock",
]
