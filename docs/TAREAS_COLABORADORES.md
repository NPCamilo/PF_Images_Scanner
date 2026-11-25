# Guía Rápida para Colaboradores — `NPCamilo/PF_Images_Scanner`

¡Bienvenido al repositorio del Proyecto Final de Comunicaciones I (UTP)!  
Todo el motor DSP se escribe en **Python (`numpy` / `scipy`)** en `dsp_engine/` y se ejecuta 100% en el navegador mediante Pyodide WebAssembly.

## 1. Configuración Inicial en tu Computador

```bash
git clone https://github.com/NPCamilo/PF_Images_Scanner.git
cd PF_Images_Scanner
pip install -r requirements.txt
npm install
npm run test:py
```

## 2. Tu Rama y Carpeta Asignada

| Integrante | Rama Git | Carpeta en `dsp_engine/` | Archivo de Pruebas |
| :--- | :--- | :--- | :--- |
| **Camilo Ramírez** | `feat/core-ui-camilo` | `dsp_engine/core/`, `bridge.py`, `src/` | `tests_py/test_core.py` |
| **Ángel Gabriel Burbano** | `feat/modulation-angel` | `dsp_engine/modulation/` | `tests_py/test_modulation.py` |
| **Luis Ospina** | `feat/demodulation-luis` | `dsp_engine/demodulation/` | `tests_py/test_demodulation.py` |
| **Mariano Orrego** | `feat/compression-docs-mariano` | `dsp_engine/compression/` + LaTeX | `tests_py/test_compression.py` |

## 3. Reglas Obligatorias (Cero Errores de Integración)
1. **Trabaja siempre en tu rama (`git checkout feat/tu-rama`).** Nunca hagas push a `main`.
2. **Respeta los contratos en `dsp_engine/contracts.py`.** Puedes importar `Signal1D`, `ModulatedSignal`, `CompressedPayload`, etc.
3. **Solo Python estándar + `numpy` + `scipy` en `dsp_engine/`:** No importes `matplotlib`, `cv2` ni `Pillow` porque el motor corre dentro del navegador vía Pyodide.
4. **Antes de abrir PR hacia `dev`:** Ejecuta `npm run verify` y confirma que `test:py`, `typecheck` y `build` pasen en verde.
