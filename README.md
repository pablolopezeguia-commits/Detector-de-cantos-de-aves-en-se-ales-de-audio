# Detector de cantos de aves en señales de audio mediante técnicas de procesado de señal aplicadas a la bioacústica


Repositorio final del TFG de Pablo López Eguía. El proyecto resuelve una tarea binaria: decidir si una ventana de audio contiene canto de ave o no. La memoria compara dos familias: una línea interpretable con rasgos acústicos manuales y SVM, y una línea profunda con embeddings YAMNet y una MLP. El modelo final se integra en EcoCanto, una herramienta de escritorio para analizar carpetas de audio y exportar resultados trazables.

## Qué contiene

```text
github/
├─ src/birdsong_detector/
│  ├─ audio.py                         # Lectura de audio y ventaneo completo
│  ├─ features.py                      # 20 rasgos acústicos finales + centroide
│  ├─ feature_selection.py             # Selección por correlación
│  ├─ augmentation.py                  # Mezcla canto + ruido de campo a SNR controlada
│  ├─ metrics.py                       # Métricas binarias comunes
│  ├─ classical/svm_pipeline.py        # Entrenamiento y evaluación SVM
│  ├─ deep_learning/yamnet_mlp.py      # Embeddings YAMNet + cabeza MLP
│  └─ ecocanto/                        # Aplicación EcoCanto
├─ scripts/                            # Puntos de entrada reproducibles
├─ configs/                            # Contratos de rasgos, inferencia y modelos
├─ models/classical_svm/               # Artefactos SVM finales serializados
├─ reports/                            # Tablas y figuras finales de la memoria
├─ data/                               # Plantilla de datos; los audios no se versionan
└─ tests/                              # Pruebas de preprocesado, segmentación, exportación y modelo
```

## Flujo del proyecto

1. Selección de características por correlación: `scripts/select_features_by_correlation.py` genera matriz de correlación, pares redundantes y heatmap.
2. Extracción de rasgos acústicos: `scripts/extract_manual_features.py` calcula los 20 rasgos finales sobre ventanas completas de 3 s con 50% de solape.
3. Línea SVM: `scripts/train_svm_family.py` entrena variantes con 20 rasgos, distancia al centroide positivo, aumento, calibración o umbral sensible.
4. Aumento de datos: `src/birdsong_detector/augmentation.py` mezcla positivos de entrenamiento con ruido de campo a SNR controlada.
5. Línea YAMNet+MLP: `scripts/train_yamnet_mlp.py` extrae embeddings medios de 1024 dimensiones y entrena la MLP final.
6. EcoCanto: `python -m birdsong_detector.ecocanto.app` abre la herramienta final con el modelo YAMNet+MLP integrado.

## Modelos incluidos

La carpeta `models/classical_svm/` incluye los cinco artefactos finales de la familia SVM: `svm_base_final.joblib`, `svm_conservative_centroid.joblib`, `svm_augmented.joblib`, `svm_calibrated.joblib` y `svm_sensitive.joblib`.

EcoCanto incluye en `src/birdsong_detector/ecocanto/model/` la cabeza `yamnet_final_mlp.keras`, el extractor YAMNet local y `metadata.json`. Por eso la aplicación puede cargar sin descargar el extractor en tiempo de uso.

La tabla `reports/tables/final_field_metrics.csv` recoge todos los modelos mencionados en la memoria. El resultado final en `TEST_CAMPO_FINAL` es:

| Modelo | Precisión | Recall | F1 | Especificidad |
| --- | ---: | ---: | ---: | ---: |
| SVM-conservador | 1.0000 | 0.6711 | 0.8032 | 1.0000 |
| SVM-aumentado | 0.7716 | 0.8389 | 0.8039 | 0.9482 |
| SVM-sensible | 0.2709 | 0.9799 | 0.4244 | 0.4496 |
| YAMNet-base | 0.8000 | 0.9933 | 0.8862 | 0.9482 |
| YAMNet-final | 0.9673 | 0.9933 | 0.9801 | 0.9930 |

## Instalación

Recomendado con Python 3.11:

```powershell
cd D:\TELECO\TFG\github
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

También se puede instalar con:

```powershell
pip install -r requirements.txt
```

## Ejecutar EcoCanto

```powershell
python -m birdsong_detector.ecocanto.app
```

La aplicación analiza una carpeta, divide los audios en ventanas completas de 3 s, calcula la probabilidad de canto con YAMNet+MLP, agrupa ventanas positivas y exporta CSV/JSON. El umbral por defecto es `0.689144`, el mismo documentado para el modelo final.

## Datos

Los audios brutos y derivados no se suben al repositorio. Hay tres motivos: tamaño, licencias de Xeno-Canto/Freefield1010 y posible contenido incidental en grabaciones de campo. La carpeta `data/` documenta el formato esperado para reconstruir ejecuciones locales.

## Comprobación rápida

```powershell
pytest -q
```

Las pruebas marcadas como `slow` cargan TensorFlow/YAMNet. Para validar solo lógica rápida:

```powershell
pytest -q -m "not slow"
```

## Figuras y trazabilidad

`reports/figures/` contiene las figuras finales de correlación, distancias a centroides, curvas ROC/PR, comparación de campo y capturas de EcoCanto. `configs/model_registry.json` enumera todos los modelos de la memoria sin usar nombres cronológicos internos.
