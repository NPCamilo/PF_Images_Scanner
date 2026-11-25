## Resumen del Cambio
- [ ] Fase / Módulo trabajado: (`core` / `modulation` / `demodulation` / `compression` / `ui`)
- [ ] Autor: (@camilo / @angel / @luis / @mariano)

## Lista de Verificación de Ingeniería (Doubt-Driven Development)
- [ ] **Contratos:** No modifiqué `src/types/dsp.ts` sin aprobación del Arquitecto Líder.
- [ ] **Precisión Numérica:** Uso `Float64Array` / `Uint8Array` sin pérdida silenciosa de precisión ni divisiones por cero sin guarda.
- [ ] **Pruebas Matemáticas:** Ejecuté `npm run test` y todas las invariantes (Parseval, BER=0 sin ruido, roundtrip lossless, SQNR) pasan en verde.
- [ ] **Compilación:** Ejecuté `npm run typecheck` y `npm run build` sin errores ni advertencias de tipos.
- [ ] **Bitácora:** Documenté fórmulas o decisiones clave para el informe LaTeX y la sustentación.
