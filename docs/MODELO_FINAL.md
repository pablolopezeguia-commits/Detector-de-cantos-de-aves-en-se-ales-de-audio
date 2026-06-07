# Ficha del modelo final

- Tarea: detección binaria canto/no canto.
- Entrada: ventanas completas de 3.0 s con 50% de solape.
- Extractor: YAMNet congelado, remuestreo a 16 kHz.
- Representación: media temporal del embedding de 1024 dimensiones.
- Clasificador: MLP con BatchNorm, Dense(256), Dropout(0.35), Dense(64), Dropout(0.25), salida sigmoide.
- Umbral operativo: `0.689144`.
- Resultado en `TEST_CAMPO_FINAL`: F1 `0.9801`, precisión `0.9673`, recall `0.9933`.
- Integración: EcoCanto.

El modelo se debe interpretar como herramienta de cribado y priorización de revisión, no como validación biológica definitiva.
