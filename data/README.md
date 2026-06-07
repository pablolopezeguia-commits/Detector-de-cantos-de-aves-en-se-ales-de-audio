# Datos esperados

Los audios no se versionan. Para reproducir entrenamiento o evaluación, prepara CSV con estas columnas mínimas:

| Columna | Descripción |
| --- | --- |
| `filepath` | Ruta al WAV/MP3/FLAC/OGG/M4A local |
| `label` | `1` para canto, `0` para no canto |
| `split` | `train`, `val`, `test` o `field_holdout` |
| `recording_id` | Identificador común para ventanas de la misma grabación |
| `sample_weight` | Opcional; peso de muestra para entrenamiento |

Los scripts escriben salidas en `outputs/`, carpeta ignorada por Git.
