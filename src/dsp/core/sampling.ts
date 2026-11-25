import { SamplingStudyResult, Signal1D } from '@/types/dsp';
import { computeAliasFrequency, computeSpectrum } from './fft';

/**
 * Simula el muestreo discreto x[n] a partir de una señal de alta resolución x(t)
 * aplicando diezmado directo sin filtro antialiasing previo para evidenciar el criterio NQS.
 */
export function simulateSamplingNQS(
  signal: Signal1D,
  downsampleFactor = 1,
  knownMaxFreqHz?: number
): SamplingStudyResult {
  const factor = Math.max(1, Math.floor(downsampleFactor));
  const originalFs = signal.fs;
  const effectiveFs = originalFs / factor;
  const effectiveTs = 1.0 / effectiveFs;

  const sampledCount = Math.ceil(signal.samples.length / factor);
  const sampledValues = new Float64Array(sampledCount);
  const sampledTimes = new Float64Array(sampledCount);

  for (let i = 0; i < sampledCount; i++) {
    const srcIdx = i * factor;
    sampledValues[i] = signal.samples[srcIdx] ?? 0;
    sampledTimes[i] = srcIdx / originalFs;
  }

  // Determinar fMax sobre la señal de referencia original antes del diezmado
  const fMaxSignalHz =
    knownMaxFreqHz ?? computeSpectrum(signal, 'hann', 0.99).fMaxHz;
  const nyquistMinimumHz = 2.0 * fMaxSignalHz;
  const nyquistSatisfied = effectiveFs >= nyquistMinimumHz;
  const nyquistMarginHz = effectiveFs - nyquistMinimumHz;

  return {
    originalFs,
    effectiveFs,
    effectiveTs,
    downsampleFactor: factor,
    sampledValues,
    sampledTimes,
    fMaxSignalHz,
    nyquistMinimumHz,
    nyquistSatisfied,
    nyquistMarginHz,
    aliasFrequencyHz: !nyquistSatisfied
      ? computeAliasFrequency(fMaxSignalHz, effectiveFs)
      : undefined,
  };
}
