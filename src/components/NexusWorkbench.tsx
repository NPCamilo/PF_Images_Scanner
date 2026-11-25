'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Binary,
  Download,
  Image as ImageIcon,
  Mic,
  Play,
  Radio,
  Sliders,
  Upload,
  Waves,
} from 'lucide-react';
import {
  captureMicrophone,
  decodeAudioFileToMono,
  playSignalAudio,
} from '@/audio/webAudioEngine';
import { BitTraceCanvas } from '@/components/canvas/BitTraceCanvas';
import { OscilloscopeCanvas } from '@/components/canvas/OscilloscopeCanvas';
import { SpectrogramCanvas } from '@/components/canvas/SpectrogramCanvas';
import { SpectrumCanvas } from '@/components/canvas/SpectrumCanvas';
import { pyodideDsp, PyodideStatus } from '@/lib/pyodideClient';
import type { LineCodingType, SyntheticWaveformType, WindowType } from '@/types/dsp';

interface AudioScanResponse {
  samples: number[];
  quantized: number[];
  errorSignal: number[];
  sampledTimes: number[];
  sampledValues: number[];
  fs: number;
  ts: number;
  duration: number;
  nqs: {
    effectiveFs: number;
    effectiveTs: number;
    nyquistMinimumHz: number;
    nyquistSatisfied: boolean;
    nyquistMarginHz: number;
    aliasFrequencyHz: number | null;
  };
  spectrum: {
    frequencies: number[];
    magnitudes: number[];
    magnitudesDb: number[];
    dominantFrequencyHz: number;
    fMaxHz: number;
    bandwidthHz: number;
  };
  spectrogram: {
    times: number[];
    frequencies: number[];
    powerDb: number[][];
  };
  quantization: {
    bitDepth: number;
    numLevels: number;
    stepSizeDelta: number;
    sqnrTheoreticalDb: number;
    sqnrMeasuredDb: number;
    totalBits: number;
    packedBytesLength: number;
    previewBits: number[];
    previewHex: string;
    bitTraceWaveform: number[];
  };
}

interface ImageScanResponse {
  width: number;
  height: number;
  originalPixels: number[];
  reconstructedPixels: number[];
  redMaskPixels: number[];
  redPixelRatio: number;
  rasterSamples: number[];
  spectrum: {
    frequencies: number[];
    magnitudesDb: number[];
    bandwidthHz: number;
  };
  metrics: {
    bitDepthPerChannel: number;
    totalBits: number;
    rawBytes: number;
    mse: number;
    psnrDb: number;
    previewBits: number[];
    previewHex: string;
  };
}

interface BitstreamScanResponse {
  bits: number[];
  totalBits: number;
  hexPreview: string;
  waveform: number[];
  fs: number;
  bitRateBps: number;
  coding: string;
  dcMean: number;
  spectrum: {
    frequencies: number[];
    magnitudesDb: number[];
    bandwidthHz: number;
    dominantFrequencyHz: number;
  };
}

function RgbaPreview({
  pixels,
  width,
  height,
  label,
}: {
  pixels: number[];
  width: number;
  height: number;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || pixels.length === 0 || width <= 0 || height <= 0) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(new Uint8ClampedArray(pixels));
    ctx.putImageData(imgData, 0, 0);
  }, [pixels, width, height]);

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-950 p-3">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-zinc-400">
        {label}
      </p>
      <canvas
        ref={ref}
        className="mx-auto aspect-square w-full max-w-[220px] rounded-lg border border-white/10 bg-black object-contain [image-rendering:pixelated]"
      />
    </div>
  );
}

export function NexusWorkbench() {
  const [pyStatus, setPyStatus] = useState<PyodideStatus>({
    status: 'idle',
    message: 'Iniciando entorno...',
  });
  const [activeTab, setActiveTab] = useState<'audio' | 'image' | 'bitstream'>('audio');
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Estado Audio / Señal x(t)
  const [waveform, setWaveform] = useState<SyntheticWaveformType>('sine');
  const [frequencyHz, setFrequencyHz] = useState(440);
  const [fs, setFs] = useState(8000);
  const [downsampleFactor, setDownsampleFactor] = useState(1);
  const [bitDepth, setBitDepth] = useState(8);
  const [windowType, setWindowType] = useState<WindowType>('hann');
  const [lineCoding, setLineCoding] = useState<LineCodingType>('polar-nrz');
  const [importedAudio, setImportedAudio] = useState<{ samples: number[]; fs: number; label: string } | null>(null);
  const [audioData, setAudioData] = useState<AudioScanResponse | null>(null);
  const [recording, setRecording] = useState(false);

  // Estado Imagen 2D
  const [imgPattern, setImgPattern] = useState<'target' | 'bars' | 'checkerboard' | 'gradient'>('target');
  const [imgBitDepth, setImgBitDepth] = useState(4);
  const [uploadedImg, setUploadedImg] = useState<{ pixels: number[]; width: number; height: number; label: string } | null>(null);
  const [imageData, setImageData] = useState<ImageScanResponse | null>(null);

  // Estado Bitstream
  const [bitSource, setBitSource] = useState<'prbs' | 'text' | 'hex'>('text');
  const [bitText, setBitText] = useState('UTP-DSP');
  const [bitHex, setBitHex] = useState('A5 5A F0 0F 3C');
  const [bitRateBps, setBitRateBps] = useState(1200);
  const [bitCoding, setBitCoding] = useState<LineCodingType>('manchester');
  const [bitData, setBitData] = useState<BitstreamScanResponse | null>(null);

  useEffect(() => {
    const unsub = pyodideDsp.onStatusChange(setPyStatus);
    void pyodideDsp.call('init').catch((err) => {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    });
    return unsub;
  }, []);

  const executeAudioScan = useCallback(async () => {
    setBusy(true);
    setErrorMsg(null);
    try {
      const res = await pyodideDsp.call<AudioScanResponse>('runAudioScan', {
        waveform,
        frequencyHz,
        fs: importedAudio ? importedAudio.fs : fs,
        durationSec: 0.08,
        downsampleFactor,
        bitDepth,
        window: windowType,
        lineCoding,
        rawSamples: importedAudio?.samples ?? null,
        label: importedAudio?.label ?? 'Sintética',
      });
      setAudioData(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [waveform, frequencyHz, fs, downsampleFactor, bitDepth, windowType, lineCoding, importedAudio]);

  const executeImageScan = useCallback(async () => {
    setBusy(true);
    setErrorMsg(null);
    try {
      const res = await pyodideDsp.call<ImageScanResponse>('runImageScan', {
        pattern: imgPattern,
        width: uploadedImg ? uploadedImg.width : 64,
        height: uploadedImg ? uploadedImg.height : 64,
        bitDepth: imgBitDepth,
        pixels: uploadedImg?.pixels ?? null,
        label: uploadedImg?.label ?? imgPattern,
      });
      setImageData(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [imgPattern, imgBitDepth, uploadedImg]);

  const executeBitstreamScan = useCallback(async () => {
    setBusy(true);
    setErrorMsg(null);
    try {
      const res = await pyodideDsp.call<BitstreamScanResponse>('runBitstreamScan', {
        source: bitSource,
        text: bitText,
        hex: bitHex,
        length: 64,
        bitRateBps,
        lineCoding: bitCoding,
        samplesPerBit: 16,
      });
      setBitData(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [bitSource, bitText, bitHex, bitRateBps, bitCoding]);

  useEffect(() => {
    if (pyStatus.status !== 'ready') return;
    if (activeTab === 'audio') void executeAudioScan();
    else if (activeTab === 'image') void executeImageScan();
    else void executeBitstreamScan();
  }, [pyStatus.status, activeTab, executeAudioScan, executeImageScan, executeBitstreamScan]);

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const decoded = await decodeAudioFileToMono(file);
      setImportedAudio({
        samples: decoded.samples.slice(0, 16384),
        fs: decoded.fs,
        label: file.name,
      });
    } catch (err) {
      setErrorMsg(`No se pudo decodificar el audio: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleMicRecord = async () => {
    setRecording(true);
    setErrorMsg(null);
    try {
      const cap = await captureMicrophone(1400);
      setImportedAudio({
        samples: cap.samples.slice(0, 16384),
        fs: cap.fs,
        label: 'Micrófono en Vivo',
      });
    } catch (err) {
      setErrorMsg(`Acceso a micrófono denegado o no disponible: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRecording(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      const maxDim = 96;
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const w = Math.max(8, Math.round(img.width * scale));
      const h = Math.max(8, Math.round(img.height * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h);
      setUploadedImg({
        pixels: Array.from(data.data),
        width: w,
        height: h,
        label: file.name,
      });
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  };

  const handleExportWav = async () => {
    if (!audioData) return;
    try {
      const b64 = await pyodideDsp.call<string>('exportWavBase64', {
        samples: audioData.quantized,
        fs: audioData.nqs.effectiveFs,
        bitDepth: 16,
      });
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const blob = new Blob([bytes], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `nexus_dsp_${audioData.quantization.bitDepth}bit.wav`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {/* Atmósfera de fondo de 3 niveles */}
      <div aria-hidden="true" className="pointer-events-none fixed inset-0">
        <div className="absolute -left-24 -top-24 h-96 w-96 rounded-full bg-cyan-500/10 blur-[120px]" />
        <div className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-emerald-500/10 blur-[120px]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* Cabecera Técnica */}
        <header className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-zinc-900/80 p-6 backdrop-blur-md lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-0.5 font-mono text-[11px] uppercase tracking-wider text-cyan-300">
                <Radio className="h-3 w-3" /> Comunicaciones I · UTP
              </span>
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[11px] ${
                  pyStatus.status === 'ready'
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                    : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                }`}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current" />
                {pyStatus.message}
              </span>
            </div>
            <h1 className="mt-2.5 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
              Nexus-DSP · Scanner de Imagen, Audio y Traza de Bits
            </h1>
            <p className="mt-1 text-xs text-zinc-400">
              Motor Python + NumPy en Pyodide WebAssembly · Fase 2 Core Interactivo
            </p>
          </div>

          <nav className="flex flex-wrap gap-2">
            {(
              [
                { id: 'audio', label: 'Audio & Señal x(t)', icon: Waves },
                { id: 'image', label: 'Imagen 2D & Escáner', icon: ImageIcon },
                { id: 'bitstream', label: 'Traza de Bits (BB)', icon: Binary },
              ] as const
            ).map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setActiveTab(id)}
                className={`inline-flex items-center gap-2 rounded-xl border px-4 py-2 text-xs font-medium transition hover:-translate-y-0.5 active:scale-[0.98] ${
                  activeTab === id
                    ? 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200'
                    : 'border-white/10 bg-zinc-950/60 text-zinc-400 hover:border-white/20 hover:text-zinc-200'
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </nav>
        </header>

        {errorMsg && (
          <div className="mt-4 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
            {errorMsg}
          </div>
        )}

        {/* TAB 1: AUDIO Y SEÑAL x(t) / x[n] */}
        {activeTab === 'audio' && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <aside className="space-y-4 rounded-2xl border border-white/10 bg-zinc-900/80 p-5 backdrop-blur-md lg:col-span-4">
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Sliders className="h-4 w-4 text-cyan-400" /> Parámetros x(t) · NQS · PCM
                </h2>
                {importedAudio && (
                  <button
                    type="button"
                    onClick={() => setImportedAudio(null)}
                    className="text-[11px] text-amber-300 underline"
                  >
                    Volver a sintética
                  </button>
                )}
              </div>

              <div className="grid grid-cols-2 gap-2">
                <label className="flex cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs text-zinc-300 hover:border-cyan-400/40">
                  <Upload className="h-3.5 w-3.5" /> Cargar WAV/MP3
                  <input type="file" accept="audio/*" onChange={handleAudioUpload} className="hidden" />
                </label>
                <button
                  type="button"
                  onClick={handleMicRecord}
                  disabled={recording}
                  className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs text-zinc-300 hover:border-rose-400/40 disabled:opacity-50"
                >
                  <Mic className="h-3.5 w-3.5 text-rose-400" />
                  {recording ? 'Grabando...' : 'Grabar 1.4s'}
                </button>
              </div>

              {!importedAudio && (
                <>
                  <div>
                    <label className="block font-mono text-[11px] uppercase text-zinc-400">
                      Forma de onda x(t)
                    </label>
                    <select
                      value={waveform}
                      onChange={(e) => setWaveform(e.target.value as SyntheticWaveformType)}
                      className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs text-zinc-100"
                    >
                      <option value="sine">Senoidal</option>
                      <option value="cosine">Cosenoidal</option>
                      <option value="square">Cuadrada</option>
                      <option value="triangle">Triangular</option>
                      <option value="sawtooth">Diente de sierra</option>
                      <option value="chirp">Chirp Lineal (Barrido)</option>
                      <option value="multitone">Multitono</option>
                      <option value="pulse-train">Tren de Pulsos</option>
                    </select>
                  </div>

                  <div>
                    <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                      <span>Frecuencia f₀</span>
                      <span className="text-cyan-300">{frequencyHz} Hz</span>
                    </div>
                    <input
                      type="range"
                      min={50}
                      max={3000}
                      step={10}
                      value={frequencyHz}
                      onChange={(e) => setFrequencyHz(Number(e.target.value))}
                      className="mt-1 w-full accent-cyan-400"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                      <span>Fs Original</span>
                      <span className="text-cyan-300">{fs} Hz</span>
                    </div>
                    <input
                      type="range"
                      min={1000}
                      max={24000}
                      step={500}
                      value={fs}
                      onChange={(e) => setFs(Number(e.target.value))}
                      className="mt-1 w-full accent-cyan-400"
                    />
                  </div>
                </>
              )}

              <div>
                <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                  <span>Submuestreo NQS (Factor M)</span>
                  <span className="text-amber-300">M={downsampleFactor}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={16}
                  step={1}
                  value={downsampleFactor}
                  onChange={(e) => setDownsampleFactor(Number(e.target.value))}
                  className="mt-1 w-full accent-amber-400"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                  <span>Resolución PCM [Bit / Byte]</span>
                  <span className="text-emerald-300">
                    {bitDepth} bits ({1 << bitDepth} niv)
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={16}
                  step={1}
                  value={bitDepth}
                  onChange={(e) => setBitDepth(Number(e.target.value))}
                  className="mt-1 w-full accent-emerald-400"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block font-mono text-[11px] uppercase text-zinc-400">
                    Ventana FFT
                  </label>
                  <select
                    value={windowType}
                    onChange={(e) => setWindowType(e.target.value as WindowType)}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
                  >
                    <option value="hann">Hann</option>
                    <option value="hamming">Hamming</option>
                    <option value="blackman">Blackman</option>
                    <option value="rectangular">Rectangular</option>
                  </select>
                </div>
                <div>
                  <label className="block font-mono text-[11px] uppercase text-zinc-400">
                    Línea BB
                  </label>
                  <select
                    value={lineCoding}
                    onChange={(e) => setLineCoding(e.target.value as LineCodingType)}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-2.5 py-2 text-xs"
                  >
                    <option value="polar-nrz">Polar NRZ</option>
                    <option value="unipolar-nrz">Unipolar NRZ</option>
                    <option value="manchester">Manchester</option>
                  </select>
                </div>
              </div>

              {audioData && (
                <div className="flex flex-wrap gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => void playSignalAudio(audioData.samples, audioData.fs)}
                    className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-cyan-500/20 px-3 py-2 text-xs font-medium text-cyan-200 hover:bg-cyan-500/30"
                  >
                    <Play className="h-3.5 w-3.5" /> Original
                  </button>
                  <button
                    type="button"
                    onClick={() => void playSignalAudio(audioData.quantized, audioData.nqs.effectiveFs)}
                    className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-amber-500/20 px-3 py-2 text-xs font-medium text-amber-200 hover:bg-amber-500/30"
                  >
                    <Play className="h-3.5 w-3.5" /> PCM {bitDepth}b
                  </button>
                  <button
                    type="button"
                    onClick={handleExportWav}
                    className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 hover:border-emerald-400/50"
                  >
                    <Download className="h-3.5 w-3.5 text-emerald-400" /> WAV
                  </button>
                </div>
              )}
            </aside>

            <section className="space-y-4 lg:col-span-8">
              {audioData ? (
                <>
                  {/* Tarjetas de Telemetría */}
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                      <p className="font-mono text-[10px] uppercase text-zinc-400">Criterio NQS</p>
                      <p
                        className={`mt-1 font-mono text-sm font-semibold ${
                          audioData.nqs.nyquistSatisfied ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {audioData.nqs.nyquistSatisfied
                          ? 'VÁLIDO'
                          : `ALIAS ${audioData.nqs.aliasFrequencyHz?.toFixed(1) ?? ''} Hz`}
                      </p>
                      <p className="font-mono text-[10px] text-zinc-500">
                        Fs_eff={audioData.nqs.effectiveFs} Hz
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                      <p className="font-mono text-[10px] uppercase text-zinc-400">SQNR Teó / Med</p>
                      <p className="mt-1 font-mono text-sm font-semibold text-amber-300">
                        {audioData.quantization.sqnrTheoreticalDb.toFixed(1)} /{' '}
                        {audioData.quantization.sqnrMeasuredDb.toFixed(1)} dB
                      </p>
                      <p className="font-mono text-[10px] text-zinc-500">
                        Δ={audioData.quantization.stepSizeDelta.toFixed(4)}
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                      <p className="font-mono text-[10px] uppercase text-zinc-400">Pico FFT / BW</p>
                      <p className="mt-1 font-mono text-sm font-semibold text-cyan-300">
                        {audioData.spectrum.dominantFrequencyHz.toFixed(1)} Hz
                      </p>
                      <p className="font-mono text-[10px] text-zinc-500">
                        BW={audioData.spectrum.bandwidthHz.toFixed(1)} Hz
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-zinc-900/80 p-3">
                      <p className="font-mono text-[10px] uppercase text-zinc-400">Trama Digital</p>
                      <p className="mt-1 font-mono text-sm font-semibold text-violet-300">
                        {audioData.quantization.totalBits} bits
                      </p>
                      <p className="font-mono text-[10px] text-zinc-500">
                        {audioData.quantization.packedBytesLength} Bytes
                      </p>
                    </div>
                  </div>

                  <OscilloscopeCanvas
                    samples={audioData.samples}
                    quantized={audioData.quantized}
                    sampledTimes={audioData.sampledTimes}
                    sampledValues={audioData.sampledValues}
                    fs={audioData.fs}
                  />

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <SpectrumCanvas
                      frequencies={audioData.spectrum.frequencies}
                      magnitudesDb={audioData.spectrum.magnitudesDb}
                      nyquistMinimumHz={audioData.nqs.nyquistMinimumHz}
                      effectiveFs={audioData.nqs.effectiveFs}
                      aliasFrequencyHz={audioData.nqs.aliasFrequencyHz}
                      bandwidthHz={audioData.spectrum.bandwidthHz}
                    />
                    <SpectrogramCanvas
                      powerDb={audioData.spectrogram.powerDb}
                      maxFreqHz={audioData.fs / 2}
                      durationSec={audioData.duration}
                    />
                  </div>

                  <BitTraceCanvas
                    waveform={audioData.quantization.bitTraceWaveform}
                    bits={audioData.quantization.previewBits}
                    coding={lineCoding}
                  />
                </>
              ) : (
                <div className="flex h-64 items-center justify-center rounded-2xl border border-white/10 bg-zinc-900/50 text-xs text-zinc-400">
                  {busy ? 'Calculando en Python/Pyodide...' : 'Esperando motor Python...'}
                </div>
              )}
            </section>
          </div>
        )}

        {/* TAB 2: IMAGEN 2D Y ESCÁNER */}
        {activeTab === 'image' && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <aside className="space-y-4 rounded-2xl border border-white/10 bg-zinc-900/80 p-5 backdrop-blur-md lg:col-span-4">
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
                  <ImageIcon className="h-4 w-4 text-cyan-400" /> Escáner Matricial 2D
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
                <Upload className="h-3.5 w-3.5" /> Subir Imagen PNG/JPG
                <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
              </label>

              {!uploadedImg && (
                <div>
                  <label className="block font-mono text-[11px] uppercase text-zinc-400">
                    Patrón de Prueba
                  </label>
                  <select
                    value={imgPattern}
                    onChange={(e) =>
                      setImgPattern(e.target.value as 'target' | 'bars' | 'checkerboard' | 'gradient')
                    }
                    className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                  >
                    <option value="target">Objetivo Concéntrico + Cruz Roja</option>
                    <option value="bars">Barras de Calibración RGB</option>
                    <option value="checkerboard">Tablero de Ajedrez (Alta Frecuencia)</option>
                    <option value="gradient">Gradiente Suave RGB</option>
                  </select>
                </div>
              )}

              <div>
                <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                  <span>Cuantización por Canal</span>
                  <span className="text-emerald-300">{imgBitDepth} bits/canal</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={imgBitDepth}
                  onChange={(e) => setImgBitDepth(Number(e.target.value))}
                  className="mt-1 w-full accent-emerald-400"
                />
              </div>

              {imageData && (
                <div className="space-y-2 rounded-xl border border-white/10 bg-zinc-950 p-3 font-mono text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Resolución:</span>
                    <span>
                      {imageData.width}×{imageData.height} px
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Trama serializada:</span>
                    <span className="text-cyan-300">{imageData.metrics.rawBytes} Bytes</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">MSE / PSNR:</span>
                    <span className="text-emerald-300">
                      {imageData.metrics.mse.toFixed(2)} / {imageData.metrics.psnrDb.toFixed(1)} dB
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Área Roja Dominante:</span>
                    <span className="text-rose-300">
                      {(imageData.redPixelRatio * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </aside>

            <section className="space-y-4 lg:col-span-8">
              {imageData ? (
                <>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <RgbaPreview
                      pixels={imageData.originalPixels}
                      width={imageData.width}
                      height={imageData.height}
                      label="Original Admitida"
                    />
                    <RgbaPreview
                      pixels={imageData.redMaskPixels}
                      width={imageData.width}
                      height={imageData.height}
                      label="Filtro Canal Rojo"
                    />
                    <RgbaPreview
                      pixels={imageData.reconstructedPixels}
                      width={imageData.width}
                      height={imageData.height}
                      label={`Reconstruida (${imgBitDepth}b/ch)`}
                    />
                  </div>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <OscilloscopeCanvas
                      samples={imageData.rasterSamples}
                      fs={imageData.width}
                      title="Barrido Raster 1D (Luminancia BT.601)"
                    />
                    <SpectrumCanvas
                      frequencies={imageData.spectrum.frequencies}
                      magnitudesDb={imageData.spectrum.magnitudesDb}
                      bandwidthHz={imageData.spectrum.bandwidthHz}
                    />
                  </div>
                </>
              ) : (
                <div className="flex h-64 items-center justify-center rounded-2xl border border-white/10 bg-zinc-900/50 text-xs text-zinc-400">
                  Escaneando matriz en Python...
                </div>
              )}
            </section>
          </div>
        )}

        {/* TAB 3: TRAZA DE BITS Y BANDA BASE */}
        {activeTab === 'bitstream' && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <aside className="space-y-4 rounded-2xl border border-white/10 bg-zinc-900/80 p-5 backdrop-blur-md lg:col-span-4">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
                <Binary className="h-4 w-4 text-violet-400" /> Generador de Trama (BB)
              </h2>

              <select
                value={bitSource}
                onChange={(e) => setBitSource(e.target.value as 'prbs' | 'text' | 'hex')}
                className="w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
              >
                <option value="text">Texto UTF-8</option>
                <option value="hex">Trama Hexadecimal</option>
                <option value="prbs">Secuencia PRBS (LFSR-16)</option>
              </select>

              {bitSource === 'text' && (
                <input
                  type="text"
                  value={bitText}
                  onChange={(e) => setBitText(e.target.value)}
                  className="w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 font-mono text-xs"
                  placeholder="Texto a serializar..."
                />
              )}

              {bitSource === 'hex' && (
                <input
                  type="text"
                  value={bitHex}
                  onChange={(e) => setBitHex(e.target.value)}
                  className="w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 font-mono text-xs"
                  placeholder="Ej: A5 5A F0 0F"
                />
              )}

              <div>
                <label className="block font-mono text-[11px] uppercase text-zinc-400">
                  Codificación de Línea
                </label>
                <select
                  value={bitCoding}
                  onChange={(e) => setBitCoding(e.target.value as LineCodingType)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                >
                  <option value="manchester">Manchester (DC = 0)</option>
                  <option value="polar-nrz">Polar NRZ-L</option>
                  <option value="unipolar-nrz">Unipolar NRZ</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between font-mono text-[11px] text-zinc-400">
                  <span>Tasa de Bits (Rb)</span>
                  <span className="text-cyan-300">{bitRateBps} bps</span>
                </div>
                <input
                  type="range"
                  min={300}
                  max={4800}
                  step={300}
                  value={bitRateBps}
                  onChange={(e) => setBitRateBps(Number(e.target.value))}
                  className="mt-1 w-full accent-violet-400"
                />
              </div>

              {bitData && (
                <div className="space-y-1.5 rounded-xl border border-white/10 bg-zinc-950 p-3 font-mono text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Bits totales:</span>
                    <span>{bitData.totalBits} bits</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Nivel DC medio:</span>
                    <span className="text-emerald-300">{bitData.dcMean.toFixed(4)} V</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-400">Ancho Banda BB:</span>
                    <span className="text-cyan-300">{bitData.spectrum.bandwidthHz.toFixed(1)} Hz</span>
                  </div>
                  <p className="break-all pt-1 text-[10px] text-zinc-500">HEX: {bitData.hexPreview}</p>
                </div>
              )}
            </aside>

            <section className="space-y-4 lg:col-span-8">
              {bitData && (
                <>
                  <BitTraceCanvas
                    waveform={bitData.waveform}
                    bits={bitData.bits}
                    coding={bitData.coding}
                  />
                  <SpectrumCanvas
                    frequencies={bitData.spectrum.frequencies}
                    magnitudesDb={bitData.spectrum.magnitudesDb}
                    bandwidthHz={bitData.spectrum.bandwidthHz}
                  />
                </>
              )}
            </section>
          </div>
        )}

        {/* Panel de Estado de Módulos del Equipo */}
        <footer className="mt-8 rounded-2xl border border-white/10 bg-zinc-900/70 p-4 backdrop-blur-md">
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2 text-zinc-300">
              <Activity className="h-4 w-4 text-emerald-400" />
              <span className="font-semibold">Estado de Integración por Ramas (`PF_Images_Scanner`):</span>
            </div>
            <div className="flex flex-wrap gap-2 font-mono text-[11px]">
              <span className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-emerald-300">
                Camilo · Core/WebAudio/Canvas (Fase 2 Lista)
              </span>
              <span className="rounded-md border border-white/10 bg-zinc-950 px-2.5 py-1 text-zinc-400">
                Ángel · Modulación CW/AWGN (Rama Lista)
              </span>
              <span className="rounded-md border border-white/10 bg-zinc-950 px-2.5 py-1 text-zinc-400">
                Luis · Demodulación/DAC/BER (Rama Lista)
              </span>
              <span className="rounded-md border border-white/10 bg-zinc-950 px-2.5 py-1 text-zinc-400">
                Mariano · Compresión/LaTeX (Rama Lista)
              </span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
