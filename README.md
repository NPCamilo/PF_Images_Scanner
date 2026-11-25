# Nexus-DSP — Scanner Unificado de Imagen, Audio y Traza de Bits ($Sw$)

**Universidad Tecnológica de Pereira (UTP) — Comunicaciones I**  
**Docente:** Ing. José Oscar Sánchez Rendón  
**Equipo:** Camilo Ramírez (Arquitecto Líder), Ángel Gabriel Burbano, Luis Ospina, Mariano Orrego.

Estación de trabajo web **100% Client-Side** (Next.js + TypeScript `Float64Array` + Web Workers + Web Audio API + HTML5 Canvas) diseñada para la adquisición, síntesis, análisis espectral (FFT/NQS), cuantización PCM, compresión (RLE/Huffman/DCT/$\mu$-law), modulación (BB/CW), simulación de canal AWGN, demodulación, reconstrucción DAC y auditoría cuantitativa de señales $x(t)$ y $x[n]$.

## Comandos Rápidos

```bash
npm install        # Instalar dependencias
npm run dev        # Servidor local de desarrollo (http://localhost:3000)
npm run typecheck  # Validación estricta TypeScript
npm run test       # Suite de pruebas matemáticas DSP en Vitest
npm run verify     # Puerta de calidad completa (typecheck + test + build)
```

## Documentación Técnica
- **SRS y Plan del Sprint:** Ver `../docs/SRS_Scanner_Imagen_Audio_Bit.md`
- **Flujo de Trabajo del Equipo:** Ver `CONTRIBUTING.md`
