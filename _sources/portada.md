# Proyecto final · Tercer entregable: Bank Marketing

```{image} imagenes/portada.jpg
:alt: Portada del proyecto final, tercer entregable: Bank Marketing
:width: 100%
```

**Modelos de Machine Learning para clasificación y regresión con optimización de hiperparámetros, optimización computacional y validación estadística**

- **Autor:** Donny Torres
- **Programa:** Maestría en Analítica de Datos, Universidad del Norte (Barranquilla)
- **Asignatura:** Machine Learning
- **Fecha:** octubre de 2026
- **Conjunto de datos:** *Bank Marketing* con contexto social y económico (`bank-additional-full.csv`), UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/222/bank+marketing> (licencia CC BY 4.0)
- **Dashboard predictivo en línea:** <https://proyecto-final-bank-marketing.onrender.com> (páginas Contexto, EDA y Modelos, con predicción en vivo y su explicación SHAP; está en el plan gratuito de Render, así que la primera carga tras 15 minutos sin visitas tarda cerca de un minuto y medio)

## Contenido

El proyecto predice, con información disponible antes de la llamada, si un cliente contactado por el telemercadeo de un banco portugués suscribirá un depósito a término, y explora como rama de regresión la duración de la llamada. Sobre el modelo base del Entregable 2 se construye el *pipeline* combinatorio que pide la guía: 7 modelos × 4 técnicas de balanceo × 4 métodos de optimización en clasificación (112), 7 modelos × 4 métodos en regresión (28) y, como extensión propia, GOAT iterativo en los modelos con tres o más hiperparámetros (25): **165 combinaciones** evaluadas con validación cruzada anidada y cronológica.

**Parte I. El dashboard y la interpretación de sus resultados.** Un capítulo por cada una de las seis pestañas del dashboard de análisis, con sus mismas figuras interactivas y una interpretación de cada resultado:

1. **Contexto y diseño experimental:** partición, explosión combinatoria y validación anidada.
2. **Tabla maestra y balanceo:** las 165 combinaciones y el efecto de SMOTE, ADASYN y `class_weight`.
3. **Optimización de hiperparámetros:** Grid, Random, Bayesiana, Genética, GOAT y Successive Halving; curvas *anytime* y eficiencia.
4. **Evaluación en prueba:** métricas con intervalos, curvas ROC y PR, matrices de confusión, calibración y regresión.
5. **Comparación estadística:** Friedman, Nemenyi, DeLong, Model Confidence Set, Giacomini-White, Clark-West y Diebold-Mariano.
6. **Interpretabilidad y costo computacional:** SHAP, LIME, complejidad teórica, escalamiento, paralelización y perfilamiento.

**Parte II. Desarrollo completo.** El cuaderno ejecutado, dividido por secciones, con el código, las tablas, las figuras y todas las interpretaciones, incluido lo que el dashboard no muestra (auditoría de calidad, EDA completo, justificación de cada método y de cada espacio de búsqueda, diagnóstico de residuos y conclusiones).

## Cómo leer las interpretaciones

Cada interpretación tiene tres párrafos: primero lo que muestra el resultado, con las cifras reales; luego lo que significa para el problema o el modelo; y al final la decisión que se toma. Las cifras se calculan en la misma ejecución que genera el texto, y el dashboard de análisis, la Parte I y la Parte II leen los mismos resultados, de modo que coinciden.

## Reproducibilidad

La partición es cronológica y, por tanto, determinista; todos los procedimientos aleatorios usan la semilla global `SEED = 42`. Las versiones de las librerías están en `requirements.txt` y cada una de las 165 combinaciones está registrada en la tabla maestra (`resultados/tabla_maestra.csv`) con sus hiperparámetros por pliegue, métricas, tiempos y semilla.
