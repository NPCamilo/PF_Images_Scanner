import { Signal1D, SpectralAnalysis, WindowType } from '@/types/dsp';

/** Retorna la menor potencia de 2 >= n */
export function nextPowerOfTwo(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Transformada Rápida de Fourier (FFT) Radix-2 iterativa in-place (Cooley-Tukey).
 * @param real Parte real (longitud debe ser potencia de 2)
 * @param imag Parte imaginaria (longitud igual a real)
 * @param inverse Si true, calcula la IFFT normalizada por 1/N
 */
export function fftRadix2(
  real: Float64Array,
  imag: Float64Array,
  inverse = false
): void {
  const n = real.length;
  if (n !== imag.length || (n & (n - 1)) !== 0 || n === 0) {
    throw new Error(`FFT Radix-2 requiere longitud potencia de 2 > 0. Recibido: ${n}`);
  }

  // Permutación bit-reversal
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = real[i]!;
      const ti = imag[i]!;
      real[i] = real[j]!;
      imag[i] = imag[j]!;
      real[j] = tr;
      imag[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Etapas mariposa (butterfly)
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= n; size <<= 1) {
    const halfSize = size >> 1;
    const angleStep = (sign * 2 * Math.PI) / size;
    const wStepR = Math.cos(angleStep);
    const wStepI = Math.sin(angleStep);

    for (let start = 0; start < n; start += size) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < halfSize; k++) {
        const evenIdx = start + k;
        const oddIdx = evenIdx + halfSize;

        const oddR = real[oddIdx]!;
        const oddI = imag[oddIdx]!;
        const tR = wR * oddR - wI * oddI;
        const tI = wR * oddI + wI * oddR;

        const evenR = real[evenIdx]!;
        const evenI = imag[evenIdx]!;

        real[evenIdx] = evenR + tR;
        imag[evenIdx] = evenI + tI;
        real[oddIdx] = evenR - tR;
        imag[oddIdx] = evenI - tI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] = real[i]! / n;
      imag[i] = imag[i]! / n;
    }
  }
}

/** Genera coeficientes de ventana espectral y retorna su ganancia coherente */
export function buildWindow(n: number, type: WindowType): { weights: Float64Array; coherentGain: number } {
  const weights = new Float64Array(n);
  if (n === 1) {
    weights[0] = 1.0;
    return { weights, coherentGain: 1.0 };
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const ratio = i / (n - 1);
    let w = 1.0;
    if (type === 'hann') {
      w = 0.5 * (1 - Math.cos(2 * Math.PI * ratio));
    } else if (type === 'hamming') {
      w = 0.54 - 0.46 * Math.cos(2 * Math.PI * ratio);
    } else if (type === 'blackman') {
      w = 0.42 - 0.5 * Math.cos(2 * Math.PI * ratio) + 0.08 * Math.cos(4 * Math.PI * ratio);
    }
    weights[i] = w;
    sum += w;
  }

  const coherentGain = sum > 0 ? sum / n : 1.0;
  return { weights, coherentGain };
}

/**
 * Calcula la frecuencia aparente reflejada por aliasing: f_alias = |f0 - round(f0/fs)*fs|
 */
export function computeAliasFrequency(signalFreqHz: number, fs: number): number {
  if (fs <= 0) return 0;
  const k = Math.round(signalFreqHz / fs);
  return Math.abs(signalFreqHz - k * fs);
}

/**
 * Análisis Espectral Unilateral calibrado con estimación de f_max, Nyquist y BW efectivo.
 */
export function computeSpectrum(
  signal: Signal1D,
  windowType: WindowType = 'hann',
  energyThreshold = 0.99,
  referenceMaxFreqHz?: number
): SpectralAnalysis {
  const n = signal.samples.length;
  const fs = signal.fs;
  if (n === 0 || fs <= 0) {
    throw new Error('Señal vacía o frecuencia de muestreo no positiva.');
  }

  const fftSize = nextPowerOfTwo(n);
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);

  const { weights, coherentGain } = buildWindow(n, windowType);
  for (let i = 0; i < n; i++) {
    real[i] = signal.samples[i]! * weights[i]!;
  }

  fftRadix2(real, imag, false);

  const halfBins = Math.floor(fftSize / 2) + 1;
  const frequencies = new Float64Array(halfBins);
  const magnitudes = new Float64Array(halfBins);
  const magnitudesDb = new Float64Array(halfBins);
  const phases = new Float64Array(halfBins);

  const normFactor = n * coherentGain;
  let peakMag = 0;
  let dominantIdx = 0;
  let totalEnergy = 0;

  for (let k = 0; k < halfBins; k++) {
    frequencies[k] = (k * fs) / fftSize;
    const re = real[k]!;
    const im = imag[k]!;
    const rawMag = Math.hypot(re, im) / normFactor;
    const oneSidedMag = k > 0 && k < halfBins - 1 ? rawMag * 2.0 : rawMag;

    magnitudes[k] = oneSidedMag;
    phases[k] = Math.atan2(im, re);

    // Ignorar DC (k=0) al buscar frecuencia dominante AC si hay energía AC
    if (k > 0 && oneSidedMag > peakMag) {
      peakMag = oneSidedMag;
      dominantIdx = k;
    }

    totalEnergy += oneSidedMag * oneSidedMag;
  }

  const maxRef = Math.max(peakMag, magnitudes[0] ?? 0, 1e-12);
  let cumulativeEnergy = 0;
  let fMaxIdx = halfBins - 1;
  let foundThreshold = false;

  for (let k = 0; k < halfBins; k++) {
    const mag = magnitudes[k]!;
    magnitudesDb[k] = 20 * Math.log10(Math.max(mag / maxRef, 1e-6));

    if (!foundThreshold && totalEnergy > 1e-15) {
      cumulativeEnergy += (mag * mag) / totalEnergy;
      if (cumulativeEnergy >= energyThreshold) {
        fMaxIdx = k;
        foundThreshold = true;
      }
    }
  }

  const fMaxMeasured = frequencies[fMaxIdx] ?? fs / 2;
  const effectiveRefMax = referenceMaxFreqHz ?? fMaxMeasured;
  const nyquistRateHz = 2.0 * effectiveRefMax;
  const isAliasing = fs < nyquistRateHz;
  const aliasFrequencyHz =
    isAliasing && referenceMaxFreqHz !== undefined
      ? computeAliasFrequency(referenceMaxFreqHz, fs)
      : undefined;

  return {
    fftSize,
    window: windowType,
    frequencies,
    magnitudes,
    magnitudesDb,
    phases,
    totalEnergy,
    fMaxHz: fMaxMeasured,
    nyquistRateHz,
    isAliasing,
    aliasFrequencyHz,
    bandwidthHz: fMaxMeasured,
    dominantFrequencyHz: frequencies[dominantIdx] ?? 0,
  };
}
