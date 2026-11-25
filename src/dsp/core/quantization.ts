import { QuantizationResult, Signal1D } from '@/types/dsp';

/**
 * Cuantización Uniforme PCM y Codificación de Resolución [Byte / Bit].
 * Soporta resoluciones N ∈ [1, 16] bits por muestra.
 */
export function quantizePCM(
  signal: Signal1D,
  bitDepth = 8,
  vMin = -1.0,
  vMax = 1.0
): QuantizationResult {
  if (!Number.isInteger(bitDepth) || bitDepth < 1 || bitDepth > 16) {
    throw new RangeError(`Resolución PCM no válida (${bitDepth}). Debe ser entero entre 1 y 16 bits.`);
  }
  if (vMax <= vMin) {
    throw new RangeError('vMax debe ser estrictamente mayor que vMin.');
  }

  const n = signal.samples.length;
  const numLevels = 1 << bitDepth; // 2^N
  const stepSizeDelta = (vMax - vMin) / (numLevels - 1);

  const quantized = new Float64Array(n);
  const levels = new Int32Array(n);
  const errorSignal = new Float64Array(n);
  const bitstream = new Uint8Array(n * bitDepth);

  let powerSignal = 0;
  let powerError = 0;

  for (let i = 0; i < n; i++) {
    const raw = signal.samples[i] ?? 0;
    const clipped = Math.min(vMax, Math.max(vMin, raw));
    const level = Math.min(
      numLevels - 1,
      Math.max(0, Math.round((clipped - vMin) / stepSizeDelta))
    );
    const qVal = vMin + level * stepSizeDelta;
    const err = clipped - qVal;

    levels[i] = level;
    quantized[i] = qVal;
    errorSignal[i] = err;

    powerSignal += clipped * clipped;
    powerError += err * err;

    // Serialización MSB -> LSB
    const bitBase = i * bitDepth;
    for (let b = 0; b < bitDepth; b++) {
      bitstream[bitBase + b] = (level >> (bitDepth - 1 - b)) & 1;
    }
  }

  const meanPowerSig = n > 0 ? powerSignal / n : 0;
  const meanPowerErr = n > 0 ? powerError / n : 0;

  const sqnrTheoreticalDb = 6.02 * bitDepth + 1.76;
  const sqnrMeasuredDb =
    meanPowerErr > 1e-12
      ? 10 * Math.log10(Math.max(meanPowerSig, 1e-12) / meanPowerErr)
      : 99.9;

  const packedBytes = packBitsToBytes(bitstream);

  return {
    quantized,
    levels,
    errorSignal,
    bitDepth,
    numLevels,
    stepSizeDelta,
    vMin,
    vMax,
    bitstream,
    packedBytes,
    sqnrTheoreticalDb,
    sqnrMeasuredDb,
  };
}

/** Empaqueta un arreglo plano de bits (0/1) en bytes (MSB primero) */
export function packBitsToBytes(bits: Uint8Array): Uint8Array {
  const numBytes = Math.ceil(bits.length / 8);
  const bytes = new Uint8Array(numBytes);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) {
      const byteIdx = i >> 3;
      const bitPos = 7 - (i & 7);
      bytes[byteIdx] = (bytes[byteIdx] ?? 0) | (1 << bitPos);
    }
  }
  return bytes;
}

/** Desempaqueta bytes a arreglo plano de bits (0/1) */
export function unpackBytesToBits(bytes: Uint8Array, totalBits?: number): Uint8Array {
  const maxBits = bytes.length * 8;
  const len = totalBits !== undefined ? Math.min(totalBits, maxBits) : maxBits;
  const bits = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    const byteIdx = i >> 3;
    const bitPos = 7 - (i & 7);
    bits[i] = ((bytes[byteIdx] ?? 0) >> bitPos) & 1;
  }
  return bits;
}

/** Reconstruye muestras PCM desde bitstream binario */
export function dequantizeBitstream(
  bitstream: Uint8Array,
  bitDepth: number,
  vMin = -1.0,
  vMax = 1.0
): Float64Array {
  const numSamples = Math.floor(bitstream.length / bitDepth);
  const numLevels = 1 << bitDepth;
  const stepSizeDelta = (vMax - vMin) / (numLevels - 1);
  const samples = new Float64Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let level = 0;
    const base = i * bitDepth;
    for (let b = 0; b < bitDepth; b++) {
      level = (level << 1) | ((bitstream[base + b] ?? 0) & 1);
    }
    samples[i] = vMin + level * stepSizeDelta;
  }
  return samples;
}
