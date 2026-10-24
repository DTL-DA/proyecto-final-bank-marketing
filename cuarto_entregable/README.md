# Proyecto final · Cuarto entregable · Bank Marketing (UCI)

Modelo original para predecir la suscripción a depósitos a término: **logística de estado latente en dos escalas (L2E)**, implementada desde cero con `numpy` y `scipy`, comparada estadísticamente contra los 140 modelos del tercer entregable. Donny Torres · Maestría en Analítica de Datos, Universidad del Norte.

## Entregables

| Entregable | Ubicación |
|---|---|
| Cuaderno completo (código, resultados e interpretaciones) | `Proyecto Final - Cuarto Entregable - Bank Marketing.ipynb` |
| Código del modelo original | `src/bm/modelo_original.py` (clase `LogisticaDosEscalas`) |
| Protocolo de validación del modelo original | `src/bm/protocolo_original.py` |
| Pruebas unitarias | `tests/test_modelo_original.py` (`pytest -q tests`) |
| Artefacto serializado y metadatos | `artifacts/l2e/` (`modelo.joblib`, `modelo_vigente.joblib`, `metadata.json`) |
| Tabla maestra de experimentos | `resultados/tabla_maestra_e4.csv` y `.parquet` |
| Revisión de literatura y bibliografía | `resultados/literatura/consultas.json` y `references.bib` |
| Cuaderno del Entregable 1, versión calificada (estado «antes» de las correcciones, sección 1.2) | `Entregable 1 - Bank Marketing.ipynb` |
| Cuaderno del Entregable 2 (evidencia de las correcciones del docente, sección 1.2) | `Entregable 2 - Bank Marketing.ipynb` |
| Integración al dashboard | `scripts/exportar_dashboard_l2e.py` (artefactos y resultados del L2E en el repositorio del dashboard) |
| Fuentes del Jupyter Book | `scripts/construir_libro.py` (capítulos 0 a 9 a partir del cuaderno ejecutado) |
| Capturas del dashboard (capítulo 7) | `imagenes/` (`scripts/capturas_dashboard.py`) |

## Diseño

- **Mismas condiciones del tercer entregable:** partición cronológica 80/20, mismo preprocesamiento, validación anidada `TimeSeriesSplit` 5 × 3 con separación de un día, semilla 42, Optuna TPE con 20 configuraciones.
- **Modelo L2E:** nivel diario latente en paseo aleatorio (escala rápida) y pendientes con deriva lenta; ajuste por moda posterior con Newton por bloques (complemento de Schur sobre un sistema tridiagonal) y filtro de Laplace con etiquetas diferidas un día.
- **Validación:** ablación de 7 variantes, Friedman y Nemenyi, Wilcoxon con Holm, t corregida de Nadeau y Bengio, prueba bayesiana de rangos con signo, DeLong, *bootstrap* estacionario y permutación por días, Diebold-Mariano, tres conjuntos adicionales de OpenML (airlines, electricity y ozone) y SHAP.

## Reproducción

```bash
conda activate bm_e3                     # entorno del tercer entregable (Python 3.12)
pip install -r requirements.txt
python scripts/importar_e3.py            # resultados del tercer entregable (benchmarks)
python scripts/ejecutar_original.py      # validación anidada, ablación, semillas y modelos finales (≈ 6 min, 10 núcleos)
python scripts/ejecutar_adicionales.py   # conjuntos adicionales (≈ 25 min)
python scripts/ejecutar_analisis.py      # sensibilidad, robustez, complejidad y SHAP
python scripts/revision_literatura.py    # revisión sistemática (APIs públicas)
python scripts/verificar_referencias.py  # BibTeX verificado por DOI
pytest -q tests
python scripts/construir_cuaderno.py --ejecutar
python scripts/exportar_dashboard_l2e.py --app ..   # en el repositorio, desde cuarto_entregable/
python scripts/construir_libro.py --libro ../book
```

## Estructura

```
data/            bank-additional-full.csv y caché de OpenML (data/adicionales)
src/bm/          paquete del proyecto (heredado del tercer entregable) + modelo_original, protocolo_original,
                 estadistica_e4 y adicionales
scripts/         experimentos, análisis, revisión de literatura y armado del cuaderno (celdas en scripts/celdas)
tests/           pruebas unitarias del modelo original
resultados/      e3 (benchmarks importados), corridas y finales del modelo original, adicionales, análisis, literatura
artifacts/l2e/   modelo serializado y metadata.json
```
