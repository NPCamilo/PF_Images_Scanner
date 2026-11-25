'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Download,
  FileJson,
  Image as ImageIcon,
  Layers,
  Play,
  Sliders,
  Upload,
} from 'lucide-react';
import { playSignalAudio } from '@/audio/webAudioEngine';
import { ConstellationCanvas } from '@/components/canvas/ConstellationCanvas';
import { EyeDiagramCanvas } from '@/components/canvas/EyeDiagramCanvas';
import { OscilloscopeCanvas } from '@/components/canvas/OscilloscopeCanvas';
import { SpectrumCanvas } from '@/components/canvas/SpectrumCanvas';
import { pyodideDsp } from '@/lib/pyodideClient';

interface CrossPipelineResponse {
  width: number;
  height: number;
  originalPixels: number[];
  reconstructedPixels: number[];
  compression: {
    codec: string;
    originalBytes: number;
    compressedBytes: number;
    compressionRatio: number;
    savingsPercent: number;
    compressedBytesHex: string;
    compressedRaw: number[];
  };
  modulation: {
    scheme: 'ASK' | 'FSK' | 'BPSK';
    carrierHz: number;
    fs: number;
    symbolRateBaud: number;
    estimatedBandwidthHz: number;
    idealConstellation: [number, number][];
    waveformPreview: number[];
    audioWaveform: number[];
    durationSec: number;
    actualSnrDb: number;
  };
  spectrum: {
    frequencies: number[];
    magnitudesDb: number[];
    dominantFrequencyHz: number;
    bandwidthHz: number;
  };
  demodulation: {
    rxConstellation: [number, number][];
    eyeTraces: number[][];
    txBitsPreview: number[];
    rxBitsPreview: number[];
  };
  metrics: {
    mse: number;
    rmse: number;
    snrDb: number;
    psnrDb: number;
    ber: number;
    bitErrors: number;
    totalBits: number;
    pearsonCorrelation: number;
  };
}

function PixelCanvasCard({
  pixels,
  width,
  height,
  title,
  canvasRefExternal,
}: {
  pixels: number[];
  width: number;
  height: number;
  title: string;
  canvasRefExternal?: React.RefObject<HTMLCanvasElement | null>;
}) {
  const localRef = useRef<HTMLCanvasElement | null>(null);
  const targetRef = canvasRefExternal ?? localRef;

  useEffect(() => {
    const canvas = targetRef.current;
    if (!canvas || pixels.length === 0) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(new Uint8ClampedArray(pixels));
    ctx.putImageData(imgData, 0, 0);
  }, [pixels, width, height, targetRef]);

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-950 p-3">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-zinc-400">
        {title}
      </p>
      <canvas
        ref={targetRef}
        className="mx-auto aspect-square w-full max-w-[190px] rounded-lg border border-white/10 bg-black [image-rendering:pixelated]"
      />
    </div>
  );
}

export function CrossPipelineTab({
  pyodideReady,
  onError,
}: {
  pyodideReady: boolean;
  onError: (msg: string | null) => void;
}) {
  const [pattern, setPattern] = useState<'target' | 'bars' | 'checkerboard' | 'gradient'>('bars');
  const [resolution, setResolution] = useState<24 | 32 | 48>(32);
  const [bitDepth, setBitDepth] = useState(4);
  const [useCompression, setUseCompression] = useState(true);
  const [modulation, setModulation] = useState<'BPSK' | 'FSK' | 'ASK'>('BPSK');
  const [snrDb, setSnrDb] = useState(60);
  const [carrierHz, setCarrierHz] = useState(1500);
  const [uploadedImg, setUploadedImg] = useState<{
    pixels: number[];
    width: number;
    height: number;
    label: string;
  } | null>(null);
  const [data, setData] = useState<CrossPipelineResponse | null>(null);
  const [busy, setBusy] = useState(false);

  const reconCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const runPipeline = useCallback(async () => {
    if (!pyodideReady) return;
    setBusy(true);
    onError(null);
    try {
      const res = await pyodideDsp.call<CrossPipelineResponse>('runCrossPipeline', {
        pattern,
        width: uploadedImg ? uploadedImg.width : resolution,
        height: uploadedImg ? uploadedImg.height : resolution,
        bitDepth,
        useCompression,
        modulation,
        snrDb,
        carrierHz,
        fs: 8000,
        samplesPerBit: 16,
        pixels: uploadedImg?.pixels ?? null,
        label: uploadedImg?.label ?? pattern,
      });
      setData(res);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [
    pyodideReady,
    pattern,
    resolution,
    bitDepth,
    useCompression,
    modulation,
    snrDb,
    carrierHz,
    uploadedImg,
    onError,
  ]);

  useEffect(() => {
    void runPipeline();
  }, [runPipeline]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      const dim = resolution;
      const canvas = document.createElement('canvas');
      canvas.width = dim;
      canvas.height = dim;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, dim, dim);
      const imgData = ctx.getImageData(0, 0, dim, dim);
      setUploadedImg({
        pixels: Array.from(imgData.data),
        width: dim,
        height: dim,
        label: file.name,
      });
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  };

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportModulatedWav = async () => {
    if (!data) return;
    const b64 = await pyodideDsp.call<string>('exportWavBase64', {
      samples: data.modulation.audioWaveform,
      fs: data.modulation.fs,
      bitDepth: 16,
    });
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    triggerDownload(
      new Blob([bytes], { type: 'audio/wav' }),
      `nexus_cw_${data.modulation.scheme}_snr${snrDb}.wav`
    );
  };

  const exportReconstructedPng = () => {
    const canvas = reconCanvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) triggerDownload(blob, `nexus_reconstructed_${modulation}.png`);
    }, 'image/png');
  };

  const exportBinaryPayload = () => {
    if (!data) return;
    const bytes = new Uint8Array(data.compression.compressedRaw);
    triggerDownload(new Blob([bytes], { type: 'application/octet-stream' }), 'nexus_frame.bin');
  };

  const exportAuditJson = () => {
    if (!data) return;
    const report = {
      timestamp: new Date().toISOString(),
      resolution: `${data.width}x${data.height}`,
      bitDepthPerChannel: bitDepth,
      compression: {
        codec: data.compression.codec,
        originalBytes: data.compression.originalBytes,
        compressedBytes: data.compression.compressedBytes,
        compressionRatio: data.compression.compressionRatio,
        savingsPercent: data.compression.savingsPercent,
      },
      modulation: {
        scheme: data.modulation.scheme,
        carrierHz: data.modulation.carrierHz,
        symbolRateBaud: data.modulation.symbolRateBaud,
        estimatedBandwidthHz: data.modulation.estimatedBandwidthHz,
        channelSnrDb: data.modulation.actualSnrDb,
      },
      auditMetrics: data.metrics,
    };
    triggerDownload(
      new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }),
      `nexus_audit_${modulation}.json`
    );
  };

  return (
    <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Controles del Pipeline Cruzado */}
      <aside className="space-y-4 rounded-2xl border border-white/10 bg-zinc-900/80 p-5 backdrop-blur-md lg:col-span-4">
        <div className="flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
            <Layers className="h-4 w-4 text-cyan-400" /> Pipeline Cruzado (Img → Audio → Img)
          </h2>
          {uploadedImg && (
            <button
              type="button"
              onClick={() => setUploadedImg(null)}
              className="text-[11px] text-amber-300 underline"
            >
              Usar patrón
            </button>
          )}
        </div>

        <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 hover:border-cyan-400/40">
          <Upload className="h-3.5 w-3.5" /> Subir Imagen Propia
          <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
        </label>

        {!uploadedImg && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block font-mono text-[11px] uppercase text-zinc-400">
                Patrón
              </label>
              <select
                value={pattern}
                onChange={(e) =>
                  setPattern(e.target.value as 'target' | 'bars' | 'checkerboard' | 'gradient')
                }
                className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
              >
                <option value="bars">Barras RGB</option>
                <option value="target">Objetivo Rojo</option>
                <option value="checkerboard">Ajedrez</option>
                <option value="gradient">Gradiente</option>
              </select>
            </div>
            <div>
              <label className="block font-mono text-[11px] uppercase text-zinc-400">
                Matriz Tx
              </label>
              <select
                value={resolution}
                onChange={(e) => setResolution(Number(e.target.value) as 24 | 32 | 48)}
                className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
              >
                <option value={24}>24 × 24 px</option>
                <option value={32}>32 × 32 px</option>
                <option value={48}>48 × 48 px</option>
              </select>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block font-mono text-[11px] uppercase text-zinc-400">
              Modulación CW
            </label>
            <select
              value={modulation}
              onChange={(e) => setModulation(e.target.value as 'BPSK' | 'FSK' | 'ASK')}
              className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
            >
              <option value="BPSK">BPSK (Coherente)</option>
              <option value="FSK">BFSK (Frecuencias)</option>
              <option value="ASK">ASK / OOK</option>
            </select>
          </div>
          <div>
            <label className="block font-mono text-[11px] uppercase text-zinc-400">
              Bits/Canal
            </label>
            <select
              value={bitDepth}
              onChange={(e) => setBitDepth(Number(e.target.value))}
              className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
            >
              <option value={2}>2 bits</option>
              <option value={4}>4 bits</option>
              <option value={8}>8 bits (Byte)</option>
            </select>
          </div>
        </div>

        <label className="flex items-center justify-between rounded-xl border border-white/10 bg-zinc-950 px-3 py-2.5 text-xs">
          <span className="flex items-center gap-2 text-zinc-200">
            <Sliders className="h-3.5 w-3.5 text-emerald-400" /> Compresión RLE
          </span>
          <input
            type="checkbox"
            checked={useCompression}
            onChange={(e) => setUseCompression(e.target.checked)}
            className="h-4 w-4 accent-emerald-400"
          />
        </label>

        <div>
          <div className="flex justify-between font-mono text-[11px] text-zinc-400">
            <span>Portadora CW (f_c)</span>
            <span className="text-cyan-300">{carrierHz} Hz</span>
          </div>
          <input
            type="range"
            min={1000}
            max={2500}
            step={250}
            value={carrierHz}
            onChange={(e) => setCarrierHz(Number(e.target.value))}
            className="mt-1 w-full accent-cyan-400"
          />
        </div>

        <div>
          <div className="flex justify-between font-mono text-[11px] text-zinc-400">
            <span>Canal AWGN (SNR)</span>
            <span className={snrDb >= 60 ? 'text-emerald-300' : 'text-amber-300'}>
              {snrDb >= 60 ? 'Ideal (∞ dB)' : `${snrDb} dB`}
            </span>
          </div>
          <input
            type="range"
            min={-12}
            max={60}
            step={1}
            value={snrDb}
            onChange={(e) => setSnrDb(Number(e.target.value))}
            className="mt-1 w-full accent-amber-400"
          />
        </div>

        {data && (
          <div className="space-y-2 pt-1">
            <button
              type="button"
              onClick={() =>
                void playSignalAudio(data.modulation.audioWaveform, data.modulation.fs)
              }
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-500/20 px-4 py-2.5 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/30"
            >
              <Play className="h-4 w-4" /> Escuchar Audio Modulado ({data.modulation.durationSec.toFixed(2)}s)
            </button>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => void exportModulatedWav()}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-[11px] text-zinc-200 hover:border-emerald-400/50"
              >
                <Download className="h-3.5 w-3.5 text-emerald-400" /> .WAV
              </button>
              <button
                type="button"
                onClick={exportReconstructedPng}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-[11px] text-zinc-200 hover:border-cyan-400/50"
              >
                <ImageIcon className="h-3.5 w-3.5 text-cyan-400" /> .PNG
              </button>
              <button
                type="button"
                onClick={exportBinaryPayload}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-[11px] text-zinc-200 hover:border-violet-400/50"
              >
                <Download className="h-3.5 w-3.5 text-violet-400" /> .BIN
              </button>
              <button
                type="button"
                onClick={exportAuditJson}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-[11px] text-zinc-200 hover:border-amber-400/50"
              >
                <FileJson className="h-3.5 w-3.5 text-amber-400" /> .JSON
              </button>
            </div>
          </div>
        )}
      </aside>

      {/* Resultados End-to-End y Auditoría */}
      <section className="space-y-4 lg:col-span-8">
        {data ? (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                <p className="font-mono text-[10px] uppercase text-zinc-400">BER (Tasa Error)</p>
                <p
                  className={`mt-1 font-mono text-sm font-semibold ${
                    data.metrics.ber === 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {(data.metrics.ber * 100).toFixed(3)}%
                </p>
                <p className="font-mono text-[10px] text-zinc-500">
                  {data.metrics.bitErrors} / {data.metrics.totalBits} bits
                </p>
              </div>
              <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                <p className="font-mono text-[10px] uppercase text-zinc-400">PSNR / Pearson ρ</p>
                <p className="mt-1 font-mono text-sm font-semibold text-cyan-300">
                  {data.metrics.psnrDb.toFixed(1)} dB · {data.metrics.pearsonCorrelation.toFixed(3)}
                </p>
                <p className="font-mono text-[10px] text-zinc-500">
                  MSE={data.metrics.mse.toFixed(2)}
                </p>
              </div>
              <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                <p className="font-mono text-[10px] uppercase text-zinc-400">Compresión ({data.compression.codec})</p>
                <p className="mt-1 font-mono text-sm font-semibold text-emerald-300">
                  {data.compression.compressionRatio.toFixed(2)}× ({data.compression.savingsPercent.toFixed(1)}%)
                </p>
                <p className="font-mono text-[10px] text-zinc-500">
                  {data.compression.originalBytes}B → {data.compression.compressedBytes}B
                </p>
              </div>
              <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                <p className="font-mono text-[10px] uppercase text-zinc-400">Canal CW / BW</p>
                <p className="mt-1 font-mono text-sm font-semibold text-amber-300">
                  {data.modulation.scheme} @ {data.modulation.carrierHz} Hz
                </p>
                <p className="font-mono text-[10px] text-zinc-500">
                  BW≈{data.modulation.estimatedBandwidthHz.toFixed(0)} Hz · {data.modulation.symbolRateBaud.toFixed(0)} Bd
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <PixelCanvasCard
                pixels={data.originalPixels}
                width={data.width}
                height={data.height}
                title={`1. Imagen Transmitida (${data.width}×${data.height})`}
              />
              <PixelCanvasCard
                pixels={data.reconstructedPixels}
                width={data.width}
                height={data.height}
                title="7. Imagen Reconstruida en RX"
                canvasRefExternal={reconCanvasRef}
              />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <OscilloscopeCanvas
                samples={data.modulation.waveformPreview}
                fs={data.modulation.fs}
                title={`Señal Audible Modulada CW (${data.modulation.scheme})`}
              />
              <SpectrumCanvas
                frequencies={data.spectrum.frequencies}
                magnitudesDb={data.spectrum.magnitudesDb}
                bandwidthHz={data.spectrum.bandwidthHz}
              />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <ConstellationCanvas
                rxPoints={data.demodulation.rxConstellation}
                idealPoints={data.modulation.idealConstellation}
                scheme={data.modulation.scheme}
              />
              <EyeDiagramCanvas
                traces={data.demodulation.eyeTraces}
                scheme={data.modulation.scheme}
              />
            </div>
          </>
        ) : (
          <div className="flex h-64 items-center justify-center rounded-2xl border border-white/10 bg-zinc-900/50 text-xs text-zinc-400">
            {busy ? 'Ejecutando Pipeline Cruzado en Python/Pyodide...' : 'Iniciando Pipeline...'}
          </div>
        )}
      </section>
    </div>
  );
}
