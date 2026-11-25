'use client';

import { useEffect, useRef } from 'react';

interface ConstellationCanvasProps {
  rxPoints: [number, number][];
  idealPoints: [number, number][];
  scheme: string;
}

export function ConstellationCanvas({
  rxPoints,
  idealPoints,
  scheme,
}: ConstellationCanvasProps) {
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

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.36;

    // Ejes I / Q y círculo unitario de referencia
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(16, cy);
    ctx.lineTo(width - 16, cy);
    ctx.moveTo(cx, 24);
    ctx.lineTo(cx, height - 16);
    ctx.stroke();

    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Puntos recibidos con ruido AWGN
    ctx.fillStyle = 'rgba(56, 189, 248, 0.55)';
    for (const [iVal, qVal] of rxPoints) {
      const x = cx + iVal * radius;
      const y = cy - qVal * radius;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Centroides ideales
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    for (const [iVal, qVal] of idealPoints) {
      const x = cx + iVal * radius;
      const y = cy - qVal * radius;
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText('Q', cx + 6, 32);
    ctx.fillText('I', width - 22, cy - 6);
    ctx.fillStyle = '#e4e4e7';
    ctx.fillText(`Constelación I/Q (${scheme}) · Amarillo=Ideal / Cian=RX`, 12, 14);
  }, [rxPoints, idealPoints, scheme]);

  return (
    <canvas
      ref={canvasRef}
      className="h-56 w-full rounded-xl border border-white/10 bg-zinc-950"
    />
  );
}
