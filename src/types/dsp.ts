/**
 * Contratos de Datos Inmutables — Nexus-DSP
 * Universidad Tecnológica de Pereira (UTP) — Comunicaciones I
 *
 * REGLA DEL SPRINT: No modificar estas interfaces sin aprobación del Arquitecto Líder.
 * Garantizan interoperabilidad entre los módulos: Core, Modulación, Demodulación y Compresión.
 */

export type DomainMode = 'audio' | 'image' | 'bitstream' | 'cross-pipeline';
export type WindowType = 'rectangular' | 'hann' | 'hamming' | 'blackman';
export type LineCodingType = 'unipolar-nrz' | 'polar-nrz' | 'manchester';
export type ModulationType =
  | 'BB'
  | 'AM-LC'
  | 'AM-SC'
  | 'FM'
  | 'ASK'
  | 'FSK'
  | 'BPSK'
  | 'QPSK';
export type CompressionType = 'none' | 'rle' | 'huffman' | 'mu-law' | 'dct-8x8';

export type SyntheticWaveformType =
  | 'sine'
  | 'cosine'
  | 'square'
  | 'triangle'
  | 'sawtooth'
  | 'chirp'
  | 'multitone'
  | 'pulse-train';

/** Parámetros para generación sintética de señales continuas/discretas */
export interface GeneratorConfig {
  waveform: SyntheticWaveformType;
  amplitude: number;
  frequencyHz: number;
  secondaryFrequencyHz?: number;
  phaseRad: number;
  dcOffset: number;
  durationSec: number;
  fs: number;
  dutyCycle?: number; // [0.05..0.95] para square/pulse-train
}

/** Señal 1D continua aproximada x(t) o discreta muestreada x[n] */
export interface Signal1D {
  id: string;
  samples: Float64Array;
  fs: number; // Frecuencia de muestreo (Hz)
  ts: number; // Periodo de muestreo 1/fs (s)
  duration: number; // Duración total (s)
  isDiscrete: boolean;
  label: string;
  metadata?: Record<string, number | string | boolean>;
}

/** Matriz de Imagen 2D normalizada para escaneo espacial/espectral */
export interface ImageMatrix {
  width: number;
  height: number;
  channels: 1 | 3 | 4; // 1: Escala de grises, 3: RGB, 4: RGBA
  pixels: Uint8ClampedArray;
  label?: string;
}

/** Resultado del Estudio de Muestreo bajo el Teorema de Nyquist-Shannon (NQS) */
export interface SamplingStudyResult {
  originalFs: number;
  effectiveFs: number;
  effectiveTs: number;
  downsampleFactor: number;
  sampledValues: Float64Array;
  sampledTimes: Float64Array;
  fMaxSignalHz: number;
  nyquistMinimumHz: number;
  nyquistSatisfied: boolean;
  nyquistMarginHz: number;
  aliasFrequencyHz?: number;
}

/** Resultado del Análisis Espectral FFT unilateral */
export interface SpectralAnalysis {
  fftSize: number;
  window: WindowType;
  frequencies: Float64Array; // Eje en Hz [0 .. fs/2]
  magnitudes: Float64Array; // |X(f)| lineal calibrado
  magnitudesDb: Float64Array; // 20*log10 normalizado (dBFS)
  phases: Float64Array; // Fase en radianes [-π, π]
  totalEnergy: number;
  fMaxHz: number; // Frecuencia máxima al 99% de energía
  nyquistRateHz: number; // 2 * fMaxHz
  isAliasing: boolean;
  aliasFrequencyHz?: number;
  bandwidthHz: number; // Ancho de banda efectivo (BW)
  dominantFrequencyHz: number;
}

/** Resultado de Cuantización PCM [Byte / Bit] */
export interface QuantizationResult {
  quantized: Float64Array; // x_q[n]
  levels: Int32Array; // Índices enteros [0 .. 2^N - 1]
  errorSignal: Float64Array; // e[n] = x[n] - x_q[n]
  bitDepth: number; // N bits por muestra (1..16)
  numLevels: number; // 2^N
  stepSizeDelta: number; // Paso Δ = (vMax - vMin) / (2^N - 1) o 2^N
  vMin: number;
  vMax: number;
  bitstream: Uint8Array; // Arreglo plano de bits individuales (0 o 1)
  packedBytes: Uint8Array; // Flujo empaquetado a resolución de Byte
  sqnrTheoreticalDb: number; // 6.02*N + 1.76
  sqnrMeasuredDb: number; // 10*log10(P_signal / P_error)
}

/** Metadatos de empaquetado para contenedor binario .dspbin */
export interface CompressionMetadata {
  sourceType: 'audio' | 'image' | 'bitstream';
  width?: number;
  height?: number;
  channels?: number;
  fs?: number;
  bitDepth?: number;
  sampleCount?: number;
  huffmanTreeJson?: string;
  quantMatrixScale?: number;
}

/** Paquete Comprimido Serializado */
export interface CompressedPayload {
  codec: CompressionType;
  originalBytes: number;
  compressedBytes: number;
  compressionRatio: number; // originalBytes / compressedBytes
  savingsPercent: number; // (1 - compressed/original) * 100
  payload: Uint8Array;
  metadata: CompressionMetadata;
}

/** Señal Modulada lista para Canal AWGN o Reproducción Web Audio */
export interface ModulatedSignal {
  scheme: ModulationType;
  lineCoding?: LineCodingType;
  waveform: Float64Array;
  fs: number; // Frecuencia de muestreo de portadora (>= 4 * fc)
  carrierHz: number; // fc (0 si es Banda Base)
  symbolRateBaud: number;
  samplesPerSymbol: number;
  estimatedBandwidthHz: number; // BW teórico (Nyquist o Carson)
  modulationIndex?: number; // m para AM o β para FM
  idealConstellation?: Array<{ i: number; q: number }>;
}

/** Salida del Canal con Ruido AWGN y Atenuación */
export interface ChannelOutput {
  noisyWaveform: Float64Array;
  noiseAdded: Float64Array;
  targetSnrDb: number;
  actualSnrDb: number;
}

/** Salida del Receptor y Demodulador */
export interface DemodulationOutput {
  recoveredSignal?: Float64Array;
  recoveredBits: Uint8Array;
  rxConstellation?: Array<{ i: number; q: number }>;
  eyeDiagramTraces?: number[][]; // Ventanas de 2 * samplesPerSymbol
}

/** Auditoría Cuantitativa de Fidelidad TX vs RX */
export interface AuditMetrics {
  mse: number;
  rmse: number;
  snrDb: number;
  psnrDb: number;
  ber: number; // Bit Error Rate [0..1]
  bitErrors: number;
  totalBits: number;
  pearsonCorrelation: number; // ρ ∈ [-1, 1]
}
