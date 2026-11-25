export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-zinc-950 px-6 py-12">
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 overflow-hidden"
      >
        <div className="absolute -left-24 -top-24 h-96 w-96 rounded-full bg-cyan-500/10 blur-[120px]" />
        <div className="absolute -bottom-24 right-0 h-96 w-96 rounded-full bg-amber-500/10 blur-[120px]" />
      </div>

      <section className="relative mx-auto max-w-5xl rounded-2xl border border-white/10 bg-zinc-900/80 p-8 backdrop-blur-md">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-cyan-400">
          UTP · Comunicaciones I · Python + Pyodide WASM Core Listo
        </p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white">
          Nexus-DSP: Scanner de Imagen, Audio y Traza de Bits
        </h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-zinc-400">
          Motor matemático en Python + NumPy ejecutado 100% en el navegador mediante
          Web Worker Pyodide (WebAssembly), con contratos congelados en{' '}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-200">
            dsp_engine/contracts.py
          </code>{' '}
          y{' '}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-200">
            src/types/dsp.ts
          </code>
          .
        </p>
      </section>
    </main>
  );
}
