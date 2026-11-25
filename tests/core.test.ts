import { describe, expect, it } from 'vitest';
import {
  bitstreamToImage,
  computeAliasFrequency,
  computeSpectrum,
  dequantizeBitstream,
  extractRedDominantMask,
  fftRadix2,
  generatePRBS,
  generateSyntheticSignal,
  generateTestImage,
  imageToBitstream,
  packBitsToBytes,
  quantizePCM,
  simulateSamplingNQS,
  unpackBytesToBits,
} from '@/dsp/core';

describe('Núcleo DSP Fase 1 — Invariantes Matemáticas (DDD)', () => {
  it('FFT Radix-2 + IFFT reconstruye exactamente la señal original (error < 1e-12)', () => {
    const n = 64;
    const origReal = new Float64Array(n);
    const real = new Float64Array(n);
    const imag = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const val = Math.sin((2 * Math.PI * 5 * i) / n) + 0.5 * Math.cos((2 * Math.PI * 13 * i) / n);
      origReal[i] = val;
      real[i] = val;
    }

    fftRadix2(real, imag, false);
    fftRadix2(real, imag, true);

    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      maxErr = Math.max(maxErr, Math.abs((real[i] ?? 0) - (origReal[i] ?? 0)));
      expect(Math.abs(imag[i] ?? 0)).toBeLessThan(1e-12);
    }
    expect(maxErr).toBeLessThan(1e-12);
  });

  it('FFT unilateral identifica frecuencia dominante exacta y cumple conservación de energía (Parseval)', () => {
    const fs = 1024;
    const freq = 100;
    const signal = generateSyntheticSignal({
      waveform: 'sine',
      amplitude: 1.0,
      frequencyHz: freq,
      phaseRad: 0,
      dcOffset: 0,
      durationSec: 1.0,
      fs,
    });

    const spec = computeSpectrum(signal, 'rectangular');
    expect(Math.abs(spec.dominantFrequencyHz - freq)).toBeLessThan(1.5);

    // Identidad de Parseval aproximada sobre magnitud unilateral en ventana rectangular
    let timeMeanPower = 0;
    for (let i = 0; i < signal.samples.length; i++) {
      const s = signal.samples[i] ?? 0;
      timeMeanPower += s * s;
    }
    timeMeanPower /= signal.samples.length;

    let freqMeanPower = (spec.magnitudes[0] ?? 0) ** 2;
    for (let k = 1; k < spec.magnitudes.length - 1; k++) {
      const mag = spec.magnitudes[k] ?? 0;
      freqMeanPower += 0.5 * mag * mag;
    }
    expect(Math.abs(freqMeanPower - timeMeanPower)).toBeLessThan(0.02);
  });

  it('Simulador NQS detecta aliasing cuando Fs < 2*f_max y calcula frecuencia reflejada exacta', () => {
    const sig = generateSyntheticSignal({
      waveform: 'sine',
      amplitude: 1,
      frequencyHz: 400,
      phaseRad: 0,
      dcOffset: 0,
      durationSec: 0.25,
      fs: 4000,
    });

    // Submuestreo por 8 -> Fs efectiva = 500 Hz (< 800 Hz Nyquist)
    const aliasingStudy = simulateSamplingNQS(sig, 8, 400);
    expect(aliasingStudy.effectiveFs).toBe(500);
    expect(aliasingStudy.nyquistMinimumHz).toBe(800);
    expect(aliasingStudy.nyquistSatisfied).toBe(false);
    expect(aliasingStudy.aliasFrequencyHz).toBe(100); // |400 - 500| = 100 Hz

    // Submuestreo por 4 -> Fs efectiva = 1000 Hz (>= 800 Hz Nyquist)
    const validStudy = simulateSamplingNQS(sig, 4, 400);
    expect(validStudy.nyquistSatisfied).toBe(true);
    expect(validStudy.aliasFrequencyHz).toBeUndefined();
    expect(computeAliasFrequency(400, 500)).toBe(100);
  });

  it('Cuantizador PCM cumple SQNR teórico ≈ 6.02N + 1.76 dB y empaquetado Byte/Bit exacto', () => {
    // Frecuencia no coherente (53.7 Hz @ 4000 Hz) para satisfacer distribución uniforme del error (hipótesis de Bennett)
    const sig = generateSyntheticSignal({
      waveform: 'sine',
      amplitude: 1.0,
      frequencyHz: 53.7,
      phaseRad: 0.13,
      dcOffset: 0,
      durationSec: 0.5,
      fs: 4000,
    });

    const pcm8 = quantizePCM(sig, 8, -1.0, 1.0);
    expect(pcm8.sqnrTheoreticalDb).toBeCloseTo(49.92, 2);
    expect(Math.abs(pcm8.sqnrMeasuredDb - pcm8.sqnrTheoreticalDb)).toBeLessThan(0.5);
    expect(pcm8.bitstream.length).toBe(sig.samples.length * 8);

    // Roundtrip empaquetado de bytes y reconstrucción de bits
    const unpacked = unpackBytesToBits(packBitsToBytes(pcm8.bitstream), pcm8.bitstream.length);
    expect(unpacked).toEqual(pcm8.bitstream);

    const decoded = dequantizeBitstream(pcm8.bitstream, 8, -1.0, 1.0);
    for (let i = 0; i < decoded.length; i++) {
      expect(decoded[i]).toBeCloseTo(pcm8.quantized[i]!, 10);
    }

    // Modo especial 1 bit
    const pcm1 = quantizePCM(sig, 1, -1.0, 1.0);
    expect(pcm1.numLevels).toBe(2);
    expect(pcm1.bitstream.length).toBe(sig.samples.length);
  });

  it('PRBS e Imagen 2D <-> Bitstream mantienen determinismo e integridad RGB', () => {
    const prbsA = generatePRBS(128, 0x1234);
    const prbsB = generatePRBS(128, 0x1234);
    expect(prbsA).toEqual(prbsB);

    const img = generateTestImage(16, 16, 'bars');
    const bits = imageToBitstream(img, 8);
    expect(bits.length).toBe(16 * 16 * 3 * 8);

    const recon = bitstreamToImage(bits, 16, 16, 3, 8);
    expect(recon.pixels).toEqual(img.pixels);

    const target = generateTestImage(32, 32, 'target');
    const redMask = extractRedDominantMask(target);
    expect(redMask.redPixelRatio).toBeGreaterThan(0.05);
  });
});
