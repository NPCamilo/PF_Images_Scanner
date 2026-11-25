"""
Contratos de Datos Inmutables — Nexus-DSP (Python Engine)
Espejo exacto de src/types/dsp.ts para garantizar interoperabilidad entre los 4 integrantes:
- Camilo Ramírez (core)
- Ángel Gabriel Burbano (modulation)
- Luis Ospina (demodulation)
- Mariano Orrego (compression)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

WindowType = Literal["rectangular", "hann", "hamming", "blackman"]
LineCodingType = Literal["unipolar-nrz", "polar-nrz", "manchester"]
ModulationType = Literal["BB", "AM-LC", "AM-SC", "FM", "ASK", "FSK", "BPSK", "QPSK"]
CompressionType = Literal["none", "rle", "huffman", "mu-law", "dct-8x8"]
SyntheticWaveformType = Literal[
    "sine", "cosine", "square", "triangle", "sawtooth", "chirp", "multitone", "pulse-train"
]


@dataclass(frozen=True)
class Signal1D:
    """Señal 1D continua aproximada x(t) o discreta muestreada x[n]."""
    id: str
    samples: NDArray[np.float64]
    fs: float
    ts: float
    duration: float
    is_discrete: bool
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageMatrix:
    """Matriz de Imagen 2D normalizada para escaneo espacial/espectral."""
    width: int
    height: int
    channels: Literal[1, 3, 4]
    pixels: NDArray[np.uint8]
    label: str = "Image"


@dataclass(frozen=True)
class SamplingStudyResult:
    """Resultado del estudio de muestreo bajo el Teorema de Nyquist-Shannon (NQS)."""
    original_fs: float
    effective_fs: float
    effective_ts: float
    downsample_factor: int
    sampled_values: NDArray[np.float64]
    sampled_times: NDArray[np.float64]
    f_max_signal_hz: float
    nyquist_minimum_hz: float
    nyquist_satisfied: bool
    nyquist_margin_hz: float
    alias_frequency_hz: Optional[float] = None


@dataclass(frozen=True)
class SpectralAnalysis:
    """Resultado del Análisis Espectral FFT unilateral."""
    fft_size: int
    window: WindowType
    frequencies: NDArray[np.float64]
    magnitudes: NDArray[np.float64]
    magnitudes_db: NDArray[np.float64]
    phases: NDArray[np.float64]
    total_energy: float
    f_max_hz: float
    nyquist_rate_hz: float
    is_aliasing: bool
    bandwidth_hz: float
    dominant_frequency_hz: float
    alias_frequency_hz: Optional[float] = None


@dataclass(frozen=True)
class QuantizationResult:
    """Resultado de Cuantización PCM [Byte / Bit]."""
    quantized: NDArray[np.float64]
    levels: NDArray[np.int32]
    error_signal: NDArray[np.float64]
    bit_depth: int
    num_levels: int
    step_size_delta: float
    v_min: float
    v_max: float
    bitstream: NDArray[np.uint8]
    packed_bytes: NDArray[np.uint8]
    sqnr_theoretical_db: float
    sqnr_measured_db: float


@dataclass(frozen=True)
class CompressedPayload:
    """Paquete Comprimido Serializado (.dspbin)."""
    codec: CompressionType
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    savings_percent: float
    payload: NDArray[np.uint8]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModulatedSignal:
    """Señal Modulada lista para Canal AWGN o Reproducción Web Audio."""
    scheme: ModulationType
    waveform: NDArray[np.float64]
    fs: float
    carrier_hz: float
    symbol_rate_baud: float
    samples_per_symbol: int
    estimated_bandwidth_hz: float
    line_coding: Optional[LineCodingType] = None
    modulation_index: Optional[float] = None
    ideal_constellation: Optional[List[Tuple[float, float]]] = None


@dataclass(frozen=True)
class ChannelOutput:
    """Salida del Canal con Ruido AWGN."""
    noisy_waveform: NDArray[np.float64]
    noise_added: NDArray[np.float64]
    target_snr_db: float
    actual_snr_db: float


@dataclass(frozen=True)
class DemodulationOutput:
    """Salida del Receptor y Demodulador."""
    recovered_bits: NDArray[np.uint8]
    recovered_signal: Optional[NDArray[np.float64]] = None
    rx_constellation: Optional[List[Tuple[float, float]]] = None
    eye_diagram_traces: Optional[List[List[float]]] = None


@dataclass(frozen=True)
class AuditMetrics:
    """Auditoría Cuantitativa de Fidelidad TX vs RX."""
    mse: float
    rmse: float
    snr_db: float
    psnr_db: float
    ber: float
    bit_errors: int
    total_bits: int
    pearson_correlation: float
