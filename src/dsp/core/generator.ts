import { GeneratorConfig, ImageMatrix, Signal1D } from '@/types/dsp';

/**
 * Genera una señal sintética continua/discreta x(t) / x[n] parametrizable.
 * Valida parámetros físicos básicos (fs > 0, durationSec > 0).
 */
export function generateSyntheticSignal(config: GeneratorConfig): Signal1D {
  const {
    waveform,
    amplitude,
    frequencyHz,
    secondaryFrequencyHz = frequencyHz * 2.5,
    phaseRad,
    dcOffset,
    durationSec,
    fs,
    dutyCycle = 0.5,
  } = config;

  if (fs <= 0 || !Number.isFinite(fs)) {
    throw new RangeError(`Frecuencia de muestreo inválida: ${fs} Hz.`);
  }
  if (durationSec <= 0 || !Number.isFinite(durationSec)) {
    throw new RangeError(`Duración inválida: ${durationSec} s.`);
  }

  const n = Math.max(1, Math.round(durationSec * fs));
  const samples = new Float64Array(n);
  const ts = 1.0 / fs;
  const clampedDuty = Math.min(0.95, Math.max(0.05, dutyCycle));

  for (let i = 0; i < n; i++) {
    const t = i * ts;
    const angle = 2 * Math.PI * frequencyHz * t + phaseRad;
    let value = 0;

    switch (waveform) {
      case 'sine':
        value = Math.sin(angle);
        break;
      case 'cosine':
        value = Math.cos(angle);
        break;
      case 'square': {
        const cyclePos = ((frequencyHz * t + phaseRad / (2 * Math.PI)) % 1 + 1) % 1;
        value = cyclePos < clampedDuty ? 1.0 : -1.0;
        break;
      }
      case 'pulse-train': {
        const cyclePos = ((frequencyHz * t + phaseRad / (2 * Math.PI)) % 1 + 1) % 1;
        value = cyclePos < clampedDuty ? 1.0 : 0.0;
        break;
      }
      case 'triangle': {
        const cyclePos = ((frequencyHz * t + phaseRad / (2 * Math.PI)) % 1 + 1) % 1;
        value = 4 * Math.abs(cyclePos - 0.5) - 1.0;
        break;
      }
      case 'sawtooth': {
        const cyclePos = ((frequencyHz * t + phaseRad / (2 * Math.PI)) % 1 + 1) % 1;
        value = 2 * cyclePos - 1.0;
        break;
      }
      case 'chirp': {
        // Barrido lineal desde frequencyHz hasta secondaryFrequencyHz
        const k = (secondaryFrequencyHz - frequencyHz) / durationSec;
        const instPhase = 2 * Math.PI * (frequencyHz * t + 0.5 * k * t * t) + phaseRad;
        value = Math.sin(instPhase);
        break;
      }
      case 'multitone': {
        const f2 = secondaryFrequencyHz;
        value = 0.6 * Math.sin(angle) + 0.4 * Math.sin(2 * Math.PI * f2 * t + phaseRad);
        break;
      }
    }

    samples[i] = dcOffset + amplitude * value;
  }

  return {
    id: `sig-${waveform}-${Date.now()}`,
    samples,
    fs,
    ts,
    duration: n * ts,
    isDiscrete: false,
    label: `${waveform.toUpperCase()} (${frequencyHz} Hz @ ${fs} Hz)`,
    metadata: {
      waveform,
      frequencyHz,
      amplitude,
      dcOffset,
    },
  };
}

/**
 * Genera una Secuencia Binaria Pseudoaleatoria (PRBS) determinística mediante LFSR de 16 bits.
 * Útil para evaluar diagramas de ojo, constelaciones y tasas BER en la Estación Bitstream.
 */
export function generatePRBS(length: number, seed = 0xace1): Uint8Array {
  if (length <= 0) {
    throw new RangeError('La longitud del bitstream PRBS debe ser mayor que 0.');
  }
  const bits = new Uint8Array(length);
  let lfsr = seed & 0xffff || 0xace1;

  for (let i = 0; i < length; i++) {
    const bit = lfsr & 1;
    bits[i] = bit;
    // Polinomio primitivo x^16 + x^14 + x^13 + x^11 + 1
    const feedback = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
    lfsr = (lfsr >> 1) | (feedback << 15);
  }
  return bits;
}

/**
 * Genera imágenes patrón sintéticas para validación sin archivos externos.
 */
export function generateTestImage(
  width: number,
  height: number,
  pattern: 'bars' | 'checkerboard' | 'gradient' | 'target'
): ImageMatrix {
  const pixels = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      let r = 0;
      let g = 0;
      let b = 0;

      if (pattern === 'bars') {
        const band = Math.floor((x / width) * 6);
        const palette: [number, number, number][] = [
          [235, 235, 235],
          [235, 200, 40],
          [40, 210, 220],
          [60, 210, 90],
          [220, 60, 200],
          [225, 50, 50],
        ];
        const color = palette[band % palette.length] ?? [255, 255, 255];
        [r, g, b] = color;
      } else if (pattern === 'checkerboard') {
        const cell = Math.max(4, Math.floor(width / 8));
        const isWhite = (Math.floor(x / cell) + Math.floor(y / cell)) % 2 === 0;
        r = g = b = isWhite ? 240 : 20;
      } else if (pattern === 'gradient') {
        r = Math.round((x / Math.max(1, width - 1)) * 255);
        g = Math.round((y / Math.max(1, height - 1)) * 255);
        b = 140;
      } else {
        // Target concéntrico + cruz roja de calibración
        const dx = x - width / 2;
        const dy = y - height / 2;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const ring = Math.sin(dist * 0.35) > 0 ? 220 : 30;
        r = ring;
        g = ring;
        b = ring;
        if (Math.abs(dx) < 3 || Math.abs(dy) < 3) {
          r = 230;
          g = 40;
          b = 40;
        }
      }

      pixels[idx] = r;
      pixels[idx + 1] = g;
      pixels[idx + 2] = b;
      pixels[idx + 3] = 255;
    }
  }

  return {
    width,
    height,
    channels: 4,
    pixels,
    label: `Pattern-${pattern}-${width}x${height}`,
  };
}
