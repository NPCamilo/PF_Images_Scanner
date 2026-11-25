/**
 * Motor Acústico del Navegador — Web Audio API
 * Permite reproducción instantánea, decodificación de WAV/MP3 y grabación en vivo por micrófono.
 */

let sharedCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (typeof window === 'undefined') {
    throw new Error('Web Audio API solo está disponible en el navegador.');
  }
  if (!sharedCtx || sharedCtx.state === 'closed') {
    sharedCtx = new window.AudioContext();
  }
  if (sharedCtx.state === 'suspended') {
    void sharedCtx.resume();
  }
  return sharedCtx;
}

export async function playSignalAudio(samples: number[], fs: number): Promise<void> {
  if (samples.length === 0 || fs <= 0) return;
  const ctx = getAudioContext();
  const safeFs = Math.min(192000, Math.max(3000, fs));
  const buffer = ctx.createBuffer(1, samples.length, safeFs);
  const channel = buffer.getChannelData(0);

  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    peak = Math.max(peak, Math.abs(samples[i] ?? 0));
  }
  const scale = peak > 1.0 ? 0.95 / peak : 0.95;
  for (let i = 0; i < samples.length; i++) {
    channel[i] = (samples[i] ?? 0) * scale;
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
}

export async function decodeAudioFileToMono(file: File): Promise<{ samples: number[]; fs: number }> {
  const ctx = getAudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  const length = decoded.length;
  const channels = decoded.numberOfChannels;
  const mono = new Array<number>(length).fill(0);

  for (let ch = 0; ch < channels; ch++) {
    const chData = decoded.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] = (mono[i] ?? 0) + (chData[i] ?? 0) / channels;
    }
  }

  return { samples: mono, fs: decoded.sampleRate };
}

export async function captureMicrophone(durationMs = 1500): Promise<{ samples: number[]; fs: number }> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = getAudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: number[] = [];

  return new Promise((resolve) => {
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < input.length; i++) {
        chunks.push(input[i] ?? 0);
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    setTimeout(() => {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      resolve({ samples: chunks, fs: ctx.sampleRate });
    }, durationMs);
  });
}
