# Flujo de Trabajo y Gobernanza — Nexus-DSP (Python + Pyodide WASM)

Este repositorio implementa el proyecto final **Scanner de Imagen - Audio - Bit (Sw)** para Comunicaciones I (UTP). El motor matemático se desarrolla **100% en Python (`numpy` / `scipy`)** y se ejecuta directamente en el navegador dentro de un **Web Worker con Pyodide (WebAssembly)** para lograr latencia cero en la sustentación.

---

## 1. Ramas por Integrante

- `main`: Producción estable (Vercel). **Prohibido push directo.**
- `dev`: Integración continua.
- Ramas personales:
  - `feat/core-ui-camilo` → Camilo Ramírez (`dsp_engine/core/`, `dsp_engine/bridge.py`, `src/`, `public/workers/`)
  - `feat/modulation-angel` → Ángel Gabriel Burbano (`dsp_engine/modulation/`)
  - `feat/demodulation-luis` → Luis Ospina (`dsp_engine/demodulation/`)
  - `feat/compression-docs-mariano` → Mariano Orrego (`dsp_engine/compression/`, Informe LaTeX)

---

## 2. Regla de Oro para Compatibilidad en Navegador (Pyodide)

1. Los módulos dentro de `dsp_engine/` deben usar únicamente librería estándar de Python + `numpy` + `scipy` (no importar `matplotlib`, `cv2`, `Pillow` ni `sounddevice` dentro de `dsp_engine/`).
2. Toda función pura recibe/retorna las estructuras congeladas de `dsp_engine/contracts.py`.
3. Toda función nueva debe tener pruebas unitarias en `tests_py/` ejecutables con:
   ```bash
   python -m unittest discover -s tests_py -v
   ```
4. Antes de abrir Pull Request hacia `dev`, ejecutar la puerta completa:
   ```bash
   npm run verify
   ```
