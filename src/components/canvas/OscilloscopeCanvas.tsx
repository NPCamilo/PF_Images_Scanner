'use client';

import { useEffect, useRef } from 'react';

interface OscilloscopeCanvasProps {
  samples: number[];
  quantized?: number[];
  sampledTimes?: number[];
  sampledValues?: number[];
  fs: number;
  title?: string;
}

export function OscilloscopeCanvas({
  samples,
  quantized,
  sampledTimes,
  sampledValues,
  fs,
  title = 'Dominio del Tiempo: x(t) y Muestras x[n]',
}: OscilloscopeCanvasProps) {
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

    // Rejilla técnica
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padT + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
    }

    // Eje 0
    const midY = padT + plotH / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.beginPath();
    ctx.moveTo(padL, midY);
    ctx.lineTo(width - padR, midY);
    ctx.stroke();

    if (samples.length === 0) return;

    let maxAbs = 1e-6;
    for (let i = 0; i < samples.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(samples[i] ?? 0));
    }
    const yScale = (plotH * 0.44) / maxAbs;

    // Decimación visual Min-Max por columna de píxel para 60 FPS
    const step = Math.max(1, Math.floor(samples.length / plotW));
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let px = 0; px < plotW; px++) {
      const idx = Math.min(samples.length - 1, Math.floor((px / plotW) * samples.length));
      let minV = samples[idx] ?? 0;
      let maxV = minV;
      for (let s = 0; s < step && idx + s < samples.length; s++) {
        const val = samples[idx + s] ?? 0;
        if (val < minV) minV = val;
        if (val > maxV) maxV = val;
      }
      const x = padL + px;
      const y1 = midY - maxV * yScale;
      const y2 = midY - minV * yScale;
      if (px === 0) ctx.moveTo(x, y1);
      else {
        ctx.lineTo(x, y1);
        if (step > 1) ctx.lineTo(x, y2);
      }
    }
    ctx.stroke();

    // Superposición PCM cuantizada (escalonada)
    if (quantized && quantized.length > 0 && quantized.length <= 600) {
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = 0; i < quantized.length; i++) {
        const x = padL + (i / Math.max(1, quantized.length - 1)) * plotW;
        const y = midY - (quantized[i] ?? 0) * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else {
          const prevY = midY - (quantized[i - 1] ?? 0) * yScale;
          ctx.lineTo(x, prevY);
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Marcadores discretos x[n]
    if (sampledTimes && sampledValues && sampledValues.length <= 220) {
      const totalDur = samples.length / Math.max(1, fs);
      ctx.fillStyle = '#fbbf24';
      for (let i = 0; i < sampledValues.length; i++) {
        const t = sampledTimes[i] ?? 0;
        const x = padL + (t / Math.max(1e-9, totalDur)) * plotW;
        const y = midY - (sampledValues[i] ?? 0) * yScale;
        ctx.beginPath();
        ctx.arc(x, y, 2.3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Etiquetas técnicas
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText(`+${maxAbs.toFixed(2)}`, 4, padT + 6);
    ctx.fillText('0.00', 4, midY + 3);
    ctx.fillText(`-${maxAbs.toFixed(2)}`, 4, padT + plotH);
    const durMs = ((samples.length / Math.max(1, fs)) * 1000).toFixed(1);
    ctx.fillText(`0 ms`, padL, height - 6);
    ctx.fillText(`${durMs} ms (Ts=${((1 / Math.max(1, fs)) * 1000).toFixed(3)} ms)`, width - 210, height - 6);
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText(title, padL, 14);
  }, [samples, quantized, sampledTimes, sampledValues, fs, title]);

  return (
    <canvas
      ref={canvasRef}
      className="h-56 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
