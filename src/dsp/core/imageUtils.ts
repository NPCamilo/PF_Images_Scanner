import { ImageMatrix } from '@/types/dsp';

/**
 * Convierte una ImageMatrix RGBA/RGB/Gray en señal 1D normalizada [-1, 1]
 * mediante barrido raster fila por fila usando luminancia ITU-R BT.601.
 */
export function imageToSignalVector(image: ImageMatrix): Float64Array {
  const totalPixels = image.width * image.height;
  const out = new Float64Array(totalPixels);
  const { channels, pixels } = image;

  for (let i = 0; i < totalPixels; i++) {
    let lum = 0;
    if (channels === 1) {
      lum = pixels[i] ?? 0;
    } else {
      const base = i * channels;
      const r = pixels[base] ?? 0;
      const g = pixels[base + 1] ?? 0;
      const b = pixels[base + 2] ?? 0;
      lum = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    out[i] = (lum / 127.5) - 1.0;
  }

  return out;
}

/** Serializa una imagen (gris o RGB) a bitstream plano (MSB primero) con resolución configurable */
export function imageToBitstream(image: ImageMatrix, bitDepth = 8): Uint8Array {
  const totalPixels = image.width * image.height;
  const outChannels = image.channels === 1 ? 1 : 3;
  const bits = new Uint8Array(totalPixels * outChannels * bitDepth);
  const maxVal = (1 << bitDepth) - 1;

  let bitCursor = 0;
  for (let i = 0; i < totalPixels; i++) {
    for (let c = 0; c < outChannels; c++) {
      const val255 =
        image.channels === 1
          ? (image.pixels[i] ?? 0)
          : (image.pixels[i * image.channels + c] ?? 0);
      const scaled = Math.round((val255 / 255) * maxVal);
      for (let b = bitDepth - 1; b >= 0; b--) {
        bits[bitCursor++] = (scaled >> b) & 1;
      }
    }
  }
  return bits;
}

/** Reconstruye una ImageMatrix RGBA desde un bitstream plano */
export function bitstreamToImage(
  bits: Uint8Array,
  width: number,
  height: number,
  sourceChannels: 1 | 3 = 3,
  bitDepth = 8
): ImageMatrix {
  const totalPixels = width * height;
  const rgba = new Uint8ClampedArray(totalPixels * 4);
  const maxVal = (1 << bitDepth) - 1;

  let bitCursor = 0;
  for (let i = 0; i < totalPixels; i++) {
    const outIdx = i * 4;
    if (sourceChannels === 1) {
      let q = 0;
      for (let b = 0; b < bitDepth; b++) {
        q = (q << 1) | (bitCursor < bits.length ? (bits[bitCursor++]! & 1) : 0);
      }
      const v = Math.round((q / Math.max(1, maxVal)) * 255);
      rgba[outIdx] = v;
      rgba[outIdx + 1] = v;
      rgba[outIdx + 2] = v;
      rgba[outIdx + 3] = 255;
    } else {
      for (let c = 0; c < 3; c++) {
        let q = 0;
        for (let b = 0; b < bitDepth; b++) {
          q = (q << 1) | (bitCursor < bits.length ? (bits[bitCursor++]! & 1) : 0);
        }
        rgba[outIdx + c] = Math.round((q / Math.max(1, maxVal)) * 255);
      }
      rgba[outIdx + 3] = 255;
    }
  }

  return {
    width,
    height,
    channels: 4,
    pixels: rgba,
    label: `Reconstructed-${width}x${height}`,
  };
}

/**
 * Filtro de color rojo dominante (reutilizado del Procesador de Imágenes DSP)
 * Identifica el componente conexo rojo predominante y genera máscara de realce.
 */
export function extractRedDominantMask(image: ImageMatrix): {
  maskedImage: ImageMatrix;
  redPixelRatio: number;
} {
  const totalPixels = image.width * image.height;
  const out = new Uint8ClampedArray(totalPixels * 4);
  let redCount = 0;

  for (let i = 0; i < totalPixels; i++) {
    const base = i * image.channels;
    const r = image.pixels[base] ?? 0;
    const g = image.channels > 1 ? (image.pixels[base + 1] ?? 0) : r;
    const b = image.channels > 2 ? (image.pixels[base + 2] ?? 0) : r;

    const isRedDominant = r > 120 && r > g * 1.45 && r > b * 1.45;
    const outBase = i * 4;
    if (isRedDominant) {
      redCount++;
      out[outBase] = r;
      out[outBase + 1] = Math.round(g * 0.25);
      out[outBase + 2] = Math.round(b * 0.25);
    } else {
      const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      const dimmed = Math.round(gray * 0.35);
      out[outBase] = dimmed;
      out[outBase + 1] = dimmed;
      out[outBase + 2] = dimmed;
    }
    out[outBase + 3] = 255;
  }

  return {
    maskedImage: {
      width: image.width,
      height: image.height,
      channels: 4,
      pixels: out,
      label: `${image.label ?? 'Image'}-RedMask`,
    },
    redPixelRatio: totalPixels > 0 ? redCount / totalPixels : 0,
  };
}
