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
