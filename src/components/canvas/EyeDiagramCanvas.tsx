'use client';

import { useEffect, useRef } from 'react';

interface EyeDiagramCanvasProps {
  traces: number[][];
  scheme: string;
}

export function EyeDiagramCanvas({ traces, scheme }: EyeDiagramCanvasProps) {
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

    const padL = 38;
    const padR = 14;
    const padT = 24;
    const padB = 22;
    const plotW = Math.max(10, width - padL - padR);
    const plotH = Math.max(10, height - padT - padB);
    const midY = padT + plotH / 2;

    // Rejilla de tiempos Tb y 2Tb
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, midY);
    ctx.lineTo(width - padR, midY);
    ctx.moveTo(padL + plotW / 2, padT);
    ctx.lineTo(padL + plotW / 2, padT + plotH);
    ctx.stroke();

    let maxAbs = 1.2;
    for (const tr of traces) {
      for (const v of tr) {
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }
    }
    const yScale = (plotH * 0.43) / maxAbs;

    ctx.strokeStyle = 'rgba(52, 211, 153, 0.35)';
    ctx.lineWidth = 1.25;
    for (const tr of traces) {
      if (tr.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < tr.length; i++) {
        const x = padL + (i / (tr.length - 1)) * plotW;
        const y = midY - (tr[i] ?? 0) * yScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText('0', padL, height - 6);
    ctx.fillText('Tb', padL + plotW / 2 - 6, height - 6);
    ctx.fillText('2Tb', width - 32, height - 6);
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText(`Diagrama de Ojo (${scheme} · ${traces.length} trazas)`, padL, 14);
  }, [traces, scheme]);

  return (
    <canvas
      ref={canvasRef}
      className="h-56 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
