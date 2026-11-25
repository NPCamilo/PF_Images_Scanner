# Flujo de Trabajo y Gobernanza — Nexus-DSP (Sprint 7 Días)

Este repositorio implementa el proyecto final **Scanner de Imagen - Audio - Bit (Sw)** para Comunicaciones I (UTP). Para minimizar errores de integración y maximizar velocidad, todo el equipo opera bajo la disciplina **Contract-First + Doubt-Driven Development (DDD)**.

---

## 1. Reglas de Ramas e Integración

- `main`: Producción estable (desplegada en Vercel). **Prohibido hacer push directo.**
- `dev`: Rama de integración continua. Solo recibe Pull Requests con CI en verde.
- Ramas personales por especialidad:
  - `feat/core-ui-camilo` → Camilo Ramírez (`src/dsp/core/`, `src/workers/`, `src/audio/`, `src/components/`)
  - `feat/modulation-angel` → Ángel Gabriel Burbano (`src/dsp/modulation/`)
  - `feat/demodulation-luis` → Luis Ospina (`src/dsp/demodulation/`)
  - `feat/compression-docs-mariano` → Mariano Orrego (`src/dsp/compression/`, actualización LaTeX)

---

## 2. Flujo de Trabajo Diario (Mínimo Margen de Error)

1. **Sincronizar antes de codificar:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout feat/tu-rama
   git merge dev
   ```
2. **Consultar el contrato congelado:** Leer `src/types/dsp.ts`. Todas las funciones puras reciben y retornan las interfaces allí definidas.
3. **Implementar con pruebas unitarias paralelas (TDD/DDD):**
   - Cada algoritmo en `src/dsp/<modulo>/` debe tener su prueba correspondiente en `tests/<modulo>.test.ts`.
   - No asumas que una fórmula funciona porque no arroja excepción: valida casos límite (señal nula `0`, frecuencia de Nyquist exacta $F_s = 2f_{\max}$, ruido extremo, bitstreams de 1 bit).
4. **Verificación Local Obligatoria antes de Push:**
   ```bash
   npm run verify
   ```
   *(Ejecuta secuencialmente `tsc --noEmit`, `vitest run` y `next build`)*
5. **Abrir Pull Request hacia `dev`:** Completar la plantilla de PR y resolver cualquier hallazgo reportado por **CodeRabbit** y GitHub Actions.
