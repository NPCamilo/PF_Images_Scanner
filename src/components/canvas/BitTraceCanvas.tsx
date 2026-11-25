'use client';

import { useEffect, useRef } from 'react';

interface BitTraceCanvasProps {
  waveform: number[];
  bits: number[];
  coding: string;
}

export function BitTraceCanvas({ waveform, bits, coding }: BitTraceCanvasProps) {
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

    const padL = 36;
    const padR = 14;
    const padT = 26;
    const padB = 22;
    const plotW = Math.max(10, width - padL - padR);
    const plotH = Math.max(10, height - padT - padB);
    const midY = padT + plotH / 2;

    const previewBitCount = Math.min(32, bits.length);
    const spb = bits.length > 0 ? Math.max(1, Math.floor(waveform.length / bits.length)) : 1;
    const maxWaveLen = previewBitCount * spb;

    if (previewBitCount > 0 && maxWaveLen > 0) {
      // Líneas verticales por bit Tb
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.font = '10px ui-monospace, monospace';
      for (let b = 0; b < previewBitCount; b++) {
        const x = padL + (b / previewBitCount) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.fillStyle = bits[b] === 1 ? '#38bdf8' : '#71717a';
        ctx.fillText(String(bits[b] ?? 0), x + 3, padT - 6);
      }

      // Forma de onda Banda Base
      ctx.strokeStyle = '#a78bfa';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const yScale = plotH * 0.4;
      for (let i = 0; i < maxWaveLen; i++) {
        const x = padL + (i / Math.max(1, maxWaveLen - 1)) * plotW;
        const y = midY - (waveform[i] ?? 0) * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('+1V', 4, padT + 10);
    ctx.fillText(' 0V', 4, midY + 3);
    ctx.fillText('-1V', 4, padT + plotH);
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText(`Traza Digital Banda Base (${coding.toUpperCase()} · primeros ${previewBitCount} bits)`, padL, 12);
  }, [waveform, bits, coding]);

  return (
    <canvas
      ref={canvasRef}
      className="h-44 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
