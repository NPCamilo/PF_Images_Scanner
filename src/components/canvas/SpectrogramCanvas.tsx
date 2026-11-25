'use client';

import { useEffect, useRef } from 'react';

interface SpectrogramCanvasProps {
  powerDb: number[][];
  maxFreqHz: number;
  durationSec: number;
}

function dbToColor(db: number): [number, number, number] {
  // Normalizar [-90, 0] dB -> [0, 1] paleta térmica de alto contraste
  const t = Math.max(0, Math.min(1, (db + 90) / 90));
  const r = Math.round(255 * Math.pow(t, 1.3));
  const g = Math.round(255 * Math.sin(t * Math.PI * 0.85) * 0.85);
  const b = Math.round(255 * (1 - t) * 0.55 * Math.sqrt(t));
  return [r, g, b];
}

export function SpectrogramCanvas({ powerDb, maxFreqHz, durationSec }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, width, height);

    const padL = 44;
    const padR = 14;
    const padT = 22;
    const padB = 24;
    const plotW = Math.max(10, width - padL - padR);
    const plotH = Math.max(10, height - padT - padB);

    const numFrames = powerDb.length;
    const numBins = powerDb[0]?.length ?? 0;
    if (numFrames > 0 && numBins > 0) {
      const cellW = plotW / numFrames;
      const cellH = plotH / numBins;
      for (let tIdx = 0; tIdx < numFrames; tIdx++) {
        const frame = powerDb[tIdx];
        if (!frame) continue;
        for (let fIdx = 0; fIdx < numBins; fIdx++) {
          const [r, g, b] = dbToColor(frame[fIdx] ?? -90);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          const x = padL + tIdx * cellW;
          const y = padT + plotH - (fIdx + 1) * cellH;
          ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText(`${maxFreqHz.toFixed(0)}Hz`, 4, padT + 8);
    ctx.fillText('0 Hz', 4, padT + plotH);
    ctx.fillText('0 s', padL, height - 6);
    ctx.fillText(`${(durationSec * 1000).toFixed(1)} ms`, width - 66, height - 6);
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText('Espectrograma STFT (Cascada Tiempo-Frecuencia)', padL, 14);
  }, [powerDb, maxFreqHz, durationSec]);

  return (
    <canvas
      ref={canvasRef}
      className="h-56 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
