/* Web Worker dedicado para ejecutar Python (NumPy/SciPy) vía Pyodide 100% Client-Side */

const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';

let pyodideReadyPromise = null;

async function initPyodideEngine() {
  self.postMessage({ type: 'status', status: 'loading-wasm', message: 'Inicializando Python WebAssembly...' });
  importScripts(`${PYODIDE_CDN}pyodide.js`);

  const pyodide = await loadPyodide({ indexURL: PYODIDE_CDN });

  self.postMessage({ type: 'status', status: 'loading-packages', message: 'Cargando NumPy en memoria local...' });
  await pyodide.loadPackage(['numpy']);

  self.postMessage({ type: 'status', status: 'loading-dsp', message: 'Montando módulos dsp_engine...' });
  const manifestRes = await fetch('/dsp_engine/manifest.json');
  if (!manifestRes.ok) {
    throw new Error('No se pudo cargar public/dsp_engine/manifest.json');
  }
  const { files } = await manifestRes.json();

  pyodide.FS.mkdirTree('/home/pyodide/dsp_engine');
  for (const relPath of files) {
    const targetPath = `/home/pyodide/dsp_engine/${relPath}`;
    const parentDir = targetPath.substring(0, targetPath.lastIndexOf('/'));
    pyodide.FS.mkdirTree(parentDir);

    const fileRes = await fetch(`/dsp_engine/${relPath}`);
    const code = await fileRes.text();
    pyodide.FS.writeFile(targetPath, code);
  }

  await pyodide.runPythonAsync(`
import sys, json
if '/home/pyodide' not in sys.path:
    sys.path.insert(0, '/home/pyodide')
import dsp_engine.bridge as bridge
  `);

  self.postMessage({ type: 'status', status: 'ready', message: 'Motor Python/NumPy listo (0 ms latencia de red).' });
  return pyodide;
}

const ACTION_MAP = {
  runCoreDemo: 'run_core_pipeline_demo',
  runAudioScan: 'run_audio_signal_scan',
  runImageScan: 'run_image_scan',
  runBitstreamScan: 'run_bitstream_scan',
  exportWavBase64: 'export_wav_base64',
};

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  try {
    if (!pyodideReadyPromise) {
      pyodideReadyPromise = initPyodideEngine();
    }
    const pyodide = await pyodideReadyPromise;

    if (action === 'init') {
      self.postMessage({ id, ok: true, result: { ready: true } });
      return;
    }

    const pyFunc = ACTION_MAP[action];
    if (!pyFunc) {
      throw new Error(`Acción desconocida en Pyodide Worker: ${action}`);
    }

    pyodide.globals.set('_js_params', pyodide.toPy(payload ?? {}));
    const jsonResult = await pyodide.runPythonAsync(
      `json.dumps(bridge.${pyFunc}(_js_params))`
    );
    self.postMessage({ id, ok: true, result: JSON.parse(jsonResult) });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
