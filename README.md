# Proyecto final · Cuarto entregable · Bank Marketing (UCI)

Proyecto final de Machine Learning. El repositorio reúne el dashboard predictivo en Dash Plotly, el modelo original del proyecto, la **logística de estado latente en dos escalas (L2E)**, y el informe técnico en Jupyter Book. El dashboard permite comprender el problema, explorar los datos y usar los modelos para predecir si un cliente suscribirá un depósito a término, con la explicación SHAP de cada predicción.

**Pregunta de investigación.** ¿Un modelo que sigue día a día la tasa de suscripción puede predecir, antes de la llamada, qué clientes suscribirán un depósito a término mejor que los clasificadores estáticos? Los clasificadores estáticos son las mejores combinaciones de 7 modelos, 4 técnicas de balanceo y 4 optimizadores (140 combinaciones con validación anidada cronológica); el L2E pasa por el mismo protocolo y los supera: AUC-PR externo de 0,111 frente a 0,103, AUC-PR de prueba de 0,602 frente a 0,548, con 3 de 4 pruebas de hipótesis a favor.

- **Dashboard en línea:** https://proyecto-final-bank-marketing.onrender.com
- **Jupyter Book del cuarto entregable:** https://dtl-da.github.io/proyecto-final-bank-marketing/cuarto-entregable/ (versión compilada en la rama `gh-pages`; el libro del tercer entregable sigue en la raíz del sitio)
- **Versión del tercer entregable:** etiqueta `tercer-entregable` de este repositorio
- **Autor:** Donny Torres · Maestría en Analítica de Datos, Universidad del Norte

## Entregables

| Entregable (guía, sección 10) | Ubicación |
|---|---|
| URL pública del dashboard | https://proyecto-final-bank-marketing.onrender.com |
| Jupyter Book (fuentes y versión publicada) | `book/` y https://dtl-da.github.io/proyecto-final-bank-marketing/cuarto-entregable/ |
| Código del modelo original con docstrings y pruebas | `src/original_model.py` y `tests/test_original_model.py` |
| Cuaderno del modelo original (código, resultados e interpretaciones) | `cuarto_entregable/Proyecto Final - Cuarto Entregable - Bank Marketing.ipynb` |
| Tabla maestra de experimentos (140 combinaciones, el L2E y sus variantes) | `cuarto_entregable/resultados/tabla_maestra_e4.csv` y `.parquet` |
| Artefactos de los modelos con `metadata.json` | `artifacts/<modelo>/` (los 7 finalistas y `l2e`) |
| Declaración de autoría y uso de herramientas de IA | Capítulo 8 del libro (sección 8.5) |
| Guion de la sustentación | `Guion de sustentación - Rúbrica - Bank Marketing.docx` |

## Secciones del dashboard

| Página | Contenido |
|---|---|
| Contexto (`/`) | Problema y relevancia con una cifra citada, pregunta de investigación, objetivos y respuesta, datos y diccionario, diagrama de la metodología, limitaciones, ética, autoría, versión y fecha. |
| EDA (`/eda`) | Filtros globales (periodo, edad, conjunto y canal) que actualizan todas las gráficas: variable objetivo, tasa mensual, explorador univariado y bivariado, correlación, faltantes y atípicos, cada una con su interpretación. |
| Modelos (`/modelos`) | Tabla de líderes con el L2E y los 7 finalistas, predicción en vivo con validación de entradas y explicación SHAP local, diagnóstico (matriz de confusión, ROC, precisión-recall y calibración), SHAP global, diagrama de diferencia crítica con el modelo original incluido, y la subsección del L2E: idea y formulación, pseudocódigo, nivel diario, convergencia de Optuna, curva de aprendizaje, diferencias con la literatura, pruebas de hipótesis con p ajustados, tamaño del efecto, ablación y veredicto. |

## Estructura

```
app.py                 punto de entrada: crea la aplicación multipágina y expone server
pages/                 contexto.py, eda.py y modelos.py (una página por sección)
src/original_model.py  clase del modelo original (copia de producción de cuarto_entregable/src/bm/modelo_original.py)
src/data.py            carga única y caché de datos, artefactos y resultados
src/plots.py           funciones que construyen las figuras (plantilla común de Plotly)
src/explicaciones.py   valores SHAP por permutaciones antitéticas
src/componentes.py     componentes propios que referencian los modelos serializados
src/bm/                paquete del tercer entregable (140 combinaciones)
artifacts/<modelo>/    modelo serializado y metadata.json (en l2e: modelo.joblib y modelo_vigente.joblib)
data/                  datos livianos, esquema de entrada y resultados precalculados
assets/estilos.css     estilos
tests/                 pruebas del dashboard con pytest (aplicación, artefactos, explicaciones y modelo original)
book/                  fuentes del Jupyter Book: _config.yml, _toc.yml, references.bib y capítulos 0 a 9
cuarto_entregable/     proyecto del modelo original: cuaderno, src/bm, scripts, pruebas, resultados y datos
scripts/               scripts del tercer entregable (experimentos, finalistas y exportación al dashboard)
```

Cada `metadata.json` guarda las versiones de las librerías, la semilla (42), los hiperparámetros finales, las columnas de entrada, las métricas del bucle externo (media y desviación estándar), las de prueba y el motivo de la selección. El esquema de entrada (nombres, tipos, rangos y categorías válidos) está en `data/esquema.json`.

## Uso local

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                   # http://127.0.0.1:8050
python -m pytest                # 56 pruebas del dashboard
```

`requirements.txt` usa `xgboost-cpu`, la misma versión de XGBoost sin las librerías de CUDA; si no está disponible en el sistema, se puede instalar `xgboost==3.4.1`.

## Reproducción del modelo original y del libro

El proyecto del modelo original está en `cuarto_entregable/` y usa el entorno `bm_e3` (Python 3.12, `cuarto_entregable/requirements.txt`). Su `README.md` describe el orden completo; en resumen, desde esa carpeta:

```bash
python scripts/ejecutar_original.py                 # validación anidada, ablación, semillas y modelos finales
python scripts/construir_cuaderno.py --ejecutar     # cuaderno ejecutado (todas las tablas y figuras)
python scripts/exportar_dashboard_l2e.py --app ..   # artefactos y resultados del L2E en el dashboard
pytest -q tests                                     # 25 pruebas unitarias del modelo original
python scripts/construir_libro.py --libro ../book   # fuentes del Jupyter Book
jupyter-book build ../book                          # Jupyter Book 1.0.4 (entorno aparte)
```

El libro no vuelve a ejecutar el cuaderno (`execute_notebooks: off`): usa sus salidas, que reproduce `construir_cuaderno.py --ejecutar`. Los resultados costosos (validación anidada, conjuntos adicionales y revisión de literatura) están versionados en `cuarto_entregable/resultados/` junto con el código que los genera.

## Decisiones de ingeniería

- **El dashboard no entrena ni optimiza.** Los pipelines se reentrenaron con todo el conjunto de entrenamiento y se serializaron con `joblib`. La prueba de ida y vuelta (`tests/test_artefactos.py`) verifica que reproducen las predicciones del cuaderno sobre 400 clientes fijos de la prueba.
- **El modelo original no depende del paquete de investigación.** `cuarto_entregable/scripts/exportar_dashboard_l2e.py` reemplaza en el artefacto las referencias a `bm` por `src/original_model.py`, cuyo algoritmo es idéntico línea por línea (lo verifica `tests/test_original_model.py`). Para predecir, el L2E usa `modelo_vigente.joblib`, que ya asimiló las etiquetas de todos los días cerrados, y asigna al cliente el día siguiente al último cerrado.
- **Explicaciones SHAP sin la librería `shap` en el servidor.** `src/explicaciones.py` implementa el mismo estimador del `PermutationExplainer` (valores de Shapley intervencionales con permutaciones antitéticas) sobre las 17 variables originales. Así la aplicación cabe en la memoria del plan gratuito de Render.
- **Un solo hilo de BLAS.** Con más hilos, KNN resuelve distinto los empates de distancia de las variables *one-hot* y sus predicciones dejarían de coincidir con las del cuaderno.
- **Probabilidades.** La SVM lineal no las produce; se escala su función de decisión con Platt. Las probabilidades de los benchmarks reflejan la tasa del entrenamiento; las del L2E siguen el régimen reciente porque su nivel diario se actualiza con los días cerrados (probabilidad media de 0,308 frente a una tasa real de 0,308 en la prueba).

## Despliegue en Render

`render.yaml` define un servicio web gratuito que instala `requirements.txt` y arranca con `gunicorn app:server`, con Python 3.12 (también fijado en `.python-version`). En Render: **New → Blueprint**, elegir este repositorio y aplicar; cada *push* a `main` dispara un despliegue nuevo. El plan gratuito suspende el servicio tras 15 minutos sin visitas: el primer acceso después tarda cerca de un minuto y medio y los siguientes responden de inmediato.

## Datos

Moro, S., Cortez, P. y Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. Datos del UCI Machine Learning Repository, licencia CC BY 4.0.
