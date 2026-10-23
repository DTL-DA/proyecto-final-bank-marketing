# Proyecto final · Tercer entregable · Bank Marketing (UCI)

Proyecto final de Machine Learning. Este repositorio contiene el dashboard predictivo en Dash Plotly, que permite comprender el problema, explorar los datos y usar los mejores modelos para predecir si un cliente suscribirá un depósito a término, con la explicación SHAP de cada predicción. Los modelos provienen de la comparación de 140 combinaciones (7 modelos, 4 técnicas de balanceo y 4 optimizadores de hiperparámetros) con validación anidada cronológica.

- **Dashboard en línea:** https://proyecto-final-bank-marketing.onrender.com
- **Jupyter Book del proyecto:** https://dtl-da.github.io/proyecto-final-bank-marketing/ (versión compilada en la rama `gh-pages`)
- **Autor:** Donny Torres · Maestría en Analítica de Datos, Universidad del Norte

## Secciones

| Página | Contenido |
|---|---|
| Contexto (`/`) | Problema y relevancia con una cifra citada, pregunta de investigación y objetivos, datos y diccionario, diagrama de la metodología, limitaciones, ética, autoría, versión y fecha. |
| EDA (`/eda`) | Filtros globales (periodo, edad, conjunto y canal) que actualizan todas las gráficas: variable objetivo, tasa mensual, explorador univariado y bivariado, correlación, faltantes y atípicos, cada una con su interpretación. |
| Modelos (`/modelos`) | Tabla de líderes ordenable y filtrable, predicción en vivo con validación de entradas y explicación SHAP local, diagnóstico (matriz de confusión, ROC, precisión-recall y calibración), SHAP global, diagrama de diferencia crítica y DeLong, y la subsección del modelo original. |

## Estructura

```
app.py                 punto de entrada: crea la aplicación y expone server
pages/                 contexto.py, eda.py y modelos.py (una página por sección)
src/data.py            carga única y caché de datos, artefactos y resultados
src/plots.py           funciones que construyen las figuras (plantilla común de Plotly)
src/explicaciones.py   valores SHAP por permutaciones antitéticas
src/componentes.py     componentes propios que referencian los modelos serializados
artifacts/<modelo>/    modelo.joblib (pipeline completo) y metadata.json
data/                  datos livianos, esquema de entrada y resultados precalculados
assets/estilos.css     estilos
tests/                 pruebas con pytest
```

Cada `metadata.json` guarda las versiones de las librerías, la semilla (42), los hiperparámetros finales, las columnas de entrada, las métricas del bucle externo (media y desviación estándar), las de prueba y el motivo de la selección. El esquema de entrada (nombres, tipos, rangos y categorías válidos) está en `data/esquema.json`.

## Uso local

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                   # http://127.0.0.1:8050
python -m pytest                # 26 pruebas
```

`requirements.txt` usa `xgboost-cpu`, la misma versión de XGBoost sin las librerías de CUDA; si no está disponible en el sistema, se puede instalar `xgboost==3.4.1`.

## Decisiones de ingeniería

- **El dashboard no entrena ni optimiza.** Los pipelines (preprocesamiento, balanceo y modelo) se reentrenaron con todo el conjunto de entrenamiento y se serializaron con `joblib`. La prueba de ida y vuelta (`tests/test_artefactos.py`) verifica que reproducen las predicciones del cuaderno sobre 400 clientes fijos de la prueba.
- **Explicaciones SHAP sin la librería `shap` en el servidor.** `src/explicaciones.py` implementa el mismo estimador del `PermutationExplainer` (valores de Shapley intervencionales con permutaciones antitéticas) sobre las 17 variables originales; cumple la propiedad de eficiencia por construcción. Así la aplicación cabe en la memoria del plan gratuito de Render (unos 320 MB al iniciar).
- **Un solo hilo de BLAS.** Con más hilos, KNN resuelve distinto los empates de distancia de las variables *one-hot* y sus predicciones dejarían de coincidir con las del cuaderno.
- **Probabilidades.** La SVM lineal no las produce; se escala su función de decisión con Platt, ajustado con los puntajes de entrenamiento. Ninguna probabilidad está recalibrada al régimen de 2009-2010, y la página lo advierte.

## Despliegue en Render

`render.yaml` define un servicio web gratuito que instala `requirements.txt` y arranca con `gunicorn app:server`, con Python 3.12 (también fijado en `.python-version`). En Render: **New → Blueprint**, elegir este repositorio y aplicar. El plan gratuito suspende el servicio tras 15 minutos sin visitas: el primer acceso después tarda cerca de un minuto y medio (93 s medidos el 21/10/2026) y los siguientes responden de inmediato.

Los artefactos se regeneran desde el proyecto del cuaderno con `scripts/exportar_dashboard.py`.

## Datos

Moro, S., Cortez, P. y Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. Datos del UCI Machine Learning Repository, licencia CC BY 4.0.
