---
name: Tarea de Fase Técnica (Nexus-DSP)
about: Registro de tarea para los colaboradores del equipo DSP (Python + Pyodide)
title: '[MÓDULO]: <Nombre de la Tarea Técnica>'
labels: ['fase-tecnica', 'python-dsp']
assignees: ''
---

### Objetivo Técnico
<!-- ¿Qué función, modulador, demodulador o algoritmo de compresión se va a implementar? -->

### Rama Asignada
- `feat/modulation-angel` (Ángel Gabriel Burbano)
- `feat/demodulation-luis` (Luis Ospina)
- `feat/compression-docs-mariano` (Mariano Orrego)

### Contratos de Entrada/Salida (`dsp_engine/contracts.py`)
<!-- Especificar dataclasses recibidos y retornados (Signal1D, ModulatedSignal, CompressedPayload, etc.) -->

### Fórmulas Matemáticas y Restricciones Pyodide
<!-- Usar únicamente librería estándar + numpy + scipy. Prohibido matplotlib/Pillow/cv2 dentro de dsp_engine/ -->

### Checklist de Definición de Terminado (DoD)
- [ ] Código modular en `dsp_engine/<tu_carpeta>/`.
- [ ] Exportación limpia en `__init__.py`.
- [ ] Mínimo 4 pruebas unitarias matemáticas en `tests_py/test_<modulo>.py`.
- [ ] `npm run verify` en verde (Python `unittest` + TypeScript + Next.js build).
- [ ] Pull Request abierto hacia `dev` para revisión de CodeRabbit y Camilo.
