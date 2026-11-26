# Nexus-DSP — Scanner Unificado de Imagen, Audio y Traza de Bits ($Sw$)

**Universidad Tecnológica de Pereira (UTP) — Comunicaciones I**  
**Docente:** Ing. José Oscar Sánchez Rendón  
**Equipo:** Camilo Ramírez (Arquitecto Líder), Ángel Gabriel Burbano, Luis Ospina, Mariano Orrego.

Estación de trabajo web **100% Client-Side con Python en el Navegador (Pyodide WebAssembly + NumPy/SciPy + Web Workers + Next.js + Web Audio API + Canvas)** para adquisición, síntesis, análisis espectral (FFT/NQS), cuantización PCM, compresión, modulación (BB/CW), canal AWGN, demodulación, reconstrucción DAC y auditoría cuantitativa.

## Comandos Rápidos

```bash
pip install -r requirements.txt  # Instalar NumPy/SciPy para pruebas locales Python
npm install                      # Instalar dependencias Next.js
npm run test:py                  # Ejecutar suite matemática en Python (unittest)
npm run dev                      # Sincronizar dsp_engine/*.py y levantar servidor web
npm run verify                   # Puerta completa: sync:py + test:py + typecheck + build
```

## Módulo de Compresión y Serialización (`dsp_engine/compression/`)

Implementación del **Bloque B (RF-06 y RF-07)** sobre `dsp_engine/contracts.py` (100% compatible con Pyodide WASM usando únicamente librería estándar de Python + `numpy`):

- **Compresión sin pérdida (`rle.py`, `huffman.py`):**
  - **RLE PackBits:** codificación y decodificación sobre bytes, traza de bits y matrices `ImageMatrix` (con soporte planar RGB opcional) garantizando $\text{MSE} = 0$ y $\text{BER} = 0\%$.
  - **Huffman:** construcción determinística de tabla de códigos prefijo, cálculo de entropía de Shannon $H(X)$, serialización de árbol en `huffmanTreeJson` y reconstrucción bit a bit exacta.
- **Compresión con pérdida (`mu_law.py`, `dct_image.py`):**
  - **$\mu$-law ($\mu = 255$):** compansión logarítmica ITU-T G.711 y cuantizador simétrico configurable ($2\dots 16$ bits) para señales `Signal1D`.
  - **DCT 2D $8 \times 8$:** transformada discreta coseno ortonormal por bloques $8 \times 8$ sobre YCbCr/escala de grises, matriz de cuantización JPEG configurable (calidad, escala o matriz personalizada) y serialización Zig-Zag + Zero-RLE con marcador EOB.
- **Contenedor `.dspbin` y exportadores (`container.py`):**
  - Empaquetado/desempaquetado binario `.dspbin` con cabecera mágica `0x44535031` (`DSP1`), esquema de modulación, $F_s$, dimensiones, metadatos JSON y verificación de integridad **CRC-32**.
  - Exportadores multi-formato: `.wav` (RIFF PCM 8/16 bits), `.png` (codificador puro sobre `zlib`), `.svg` vectorial, `.bin` y reporte de sesión `.json`.
- **Pruebas unitarias (`tests_py/test_compression.py`):** verificación matemática de invariantes (roundtrip sin pérdida, teorema de Shannon, ganancia SQNR $\mu$-law, ortonormalidad DCT, integridad `.dspbin` y exportadores) ejecutable con `python -m unittest tests_py/test_compression.py -v`.
