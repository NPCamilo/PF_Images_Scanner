'use client';

import { useEffect, useRef } from 'react';

interface SpectrumCanvasProps {
  frequencies: number[];
  magnitudesDb: number[];
  nyquistMinimumHz?: number;
  effectiveFs?: number;
  aliasFrequencyHz?: number | null;
  bandwidthHz?: number;
}

export function SpectrumCanvas({
  frequencies,
  magnitudesDb,
  nyquistMinimumHz,
  effectiveFs,
  aliasFrequencyHz,
  bandwidthHz,
}: SpectrumCanvasProps) {
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

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padT + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
    }

    if (frequencies.length === 0 || magnitudesDb.length === 0) return;
    const maxFreq = frequencies[frequencies.length - 1] || 1;

    // Región sombreada de Ancho de Banda (BW)
    if (bandwidthHz && bandwidthHz > 0) {
      const bwX = padL + Math.min(1, bandwidthHz / maxFreq) * plotW;
      ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
      ctx.fillRect(padL, padT, bwX - padL, plotH);
    }

    // Curva FFT (dBFS en rango [-120, 0])
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < frequencies.length; i++) {
      const f = frequencies[i] ?? 0;
      const db = Math.max(-120, Math.min(0, magnitudesDb[i] ?? -120));
      const x = padL + (f / maxFreq) * plotW;
      const y = padT + ((-db) / 120) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Marcador Nyquist Fs_eff/2
    if (effectiveFs && effectiveFs / 2 <= maxFreq) {
      const nyqX = padL + ((effectiveFs / 2) / maxFreq) * plotW;
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.8)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(nyqX, padT);
      ctx.lineTo(nyqX, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Marcador de Aliasing
    if (aliasFrequencyHz !== null && aliasFrequencyHz !== undefined && aliasFrequencyHz <= maxFreq) {
      const alX = padL + (aliasFrequencyHz / maxFreq) * plotW;
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(alX, padT);
      ctx.lineTo(alX, padT + plotH);
      ctx.stroke();
      ctx.fillStyle = '#fb7185';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(`ALIAS ${aliasFrequencyHz.toFixed(1)} Hz`, Math.min(width - 110, alX + 4), padT + 16);
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText('0 dB', 4, padT + 6);
    ctx.fillText('-60 dB', 4, padT + plotH / 2);
    ctx.fillText('-120', 4, padT + plotH);
    ctx.fillText('0 Hz', padL, height - 6);
    ctx.fillText(`${maxFreq.toFixed(0)} Hz`, width - 58, height - 6);
    ctx.fillStyle = '#e4e4e7';
    const nyqLabel = nyquistMinimumHz ? ` · NQS min=${nyquistMinimumHz.toFixed(0)} Hz` : '';
    ctx.fillText(`Espectro FFT |X(f)|${nyqLabel}`, padL, 14);
  }, [frequencies, magnitudesDb, nyquistMinimumHz, effectiveFs, aliasFrequencyHz, bandwidthHz]);

  return (
    <canvas
      ref={canvasRef}
      className="h-56 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
