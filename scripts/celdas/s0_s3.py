"""Celdas nuevas: encabezado, configuración del proyecto y nuevos hallazgos del EDA (sección 3.8)."""

ENCABEZADO = r'''# Proyecto Final · Tercer entregable · Predicción de la suscripción a depósitos a término (*Bank Marketing*, UCI)

**Modelos de Machine Learning para clasificación y regresión: optimización de hiperparámetros, optimización computacional y validación estadística**

| | |
|---|---|
| **Autor** | Donny Torres |
| **Programa** | Maestría en Analítica de Datos, Universidad del Norte (Barranquilla) |
| **Asignatura** | Machine Learning |
| **Fecha** | Octubre de 2026 |
| **Conjunto de datos** | *Bank Marketing* con contexto social y económico (`bank-additional-full.csv`) |
| **Fuente** | UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/222/bank+marketing> (licencia CC BY 4.0) |
| **Tareas** | Clasificación de `y` (suscribe o no) y regresión de `log1p(duration)` (duración de la llamada), ambas con información previa a la llamada y validación cronológica |

Este cuaderno desarrolla el tercer entregable del proyecto final y continúa el trabajo del Entregable 2: mismo conjunto de datos, misma partición cronológica 80/20, mismo preprocesamiento y el mismo modelo base (regresión logística con C = 0,01) como referencia. Sobre esa base construye el *pipeline* combinatorio que pide la guía: 7 modelos de clasificación × 4 técnicas de balanceo × 4 métodos de optimización (112) y 7 modelos de regresión × 4 métodos (28), para un total de **140 combinaciones** evaluadas con **validación cruzada anidada**.

**Organización.** Las secciones 1 a 3 conservan la descripción y la auditoría de calidad de la base de datos, la reserva del conjunto de prueba, el análisis exploratorio y el preprocesamiento del Entregable 2, que ya incorporan las correcciones de la primera entrega (componente temporal completo, regularización elegida por pérdida logarítmica, estabilidad de coeficientes y recalibración mensual del intercepto); la sección 3.8 agrega los hallazgos nuevos que motivan las decisiones de este entregable. Las referencias del tipo "enunciado, sección x" de esas secciones remiten a la guía del primer entregable. Las secciones nuevas siguen la guía técnica de este entregable (*Modelos de Machine Learning para Clasificación y Regresión*):

| Sección del cuaderno | Contenido | Sección de la guía |
|---|---|---|
| 4. Diseño experimental | Explosión combinatoria, validación anidada, métrica de validación y reproducibilidad | 2, 3.2, 5.1.1 y 7.3 |
| 5. Optimización de hiperparámetros | Espacios de búsqueda, Grid, Random, Bayesiana, Genética, multi-fidelidad, curvas *anytime* y semillas | 3 |
| 6. Optimización computacional | Complejidad teórica frente a tiempos medidos, paralelización y perfilamiento | 4 |
| 7. Evaluación de modelos | Tabla maestra, balanceo, métricas en prueba, calibración, residuos, SHAP y LIME, robustez | 5 |
| 8. Comparación estadística | Friedman, Nemenyi y diagrama CD, DeLong, MCS, Giacomini-White, Clark-West, Diebold-Mariano, *bootstrap* y tamaño de efecto | 6 |
| 9 y 10 | Conclusiones, limitaciones y referencias | 9 |

**Ejecución.** Las 140 combinaciones con validación anidada implican decenas de miles de ajustes de modelos, así que se ejecutan con los *scripts* de la carpeta `scripts/` (en paralelo y con registro por tarea en `resultados/corridas/`) y este cuaderno lee sus resultados. Con `RECALCULAR = True` en la celda de configuración del proyecto el cuaderno vuelve a lanzar esos *scripts*. Todas las interpretaciones se generan con los valores calculados en la misma ejecución, de forma que las cifras del texto coinciden siempre con las salidas.'''

CONFIG_MD = r'''## Configuración del entorno y reproducibilidad

La partición entrenamiento-prueba es cronológica y, por tanto, determinista. Para todo procedimiento aleatorio se fija una **semilla global única** (`SEED = 42`), definida en `src/bm/config.py` y propagada de forma explícita a `numpy`, scikit-learn (`random_state` de cada estimador), XGBoost, SMOTE y ADASYN, Optuna (`TPESampler(seed=...)`) y DEAP (módulo `random`). El entorno es `bm_e3` (Python 3.12) y las versiones exactas de todas las librerías están congeladas en `requirements.txt`; la tabla siguiente reporta las del análisis exploratorio, y la celda de módulos del tercer entregable (antes de la sección 3.8), las que se agregan en este entregable.'''

CONFIG_PROYECTO = r'''import json
import pickle
import subprocess
import sys

sys.path.insert(0, str(Path("src").resolve()))

import arch
import deap
import faiss
import imblearn
import lime
import optuna
import scikit_posthocs as sp
import shap
import xgboost
from scipy.special import expit
from sklearn.metrics import (ConfusionMatrixDisplay, mean_absolute_error, precision_recall_curve as curva_pr,
                             r2_score, root_mean_squared_error)

from statsmodels.tsa.stattools import acf

from bm import computacional as comp
from bm import config as cfg
from bm import estadistica as st
from bm import evaluacion as evm
from bm.anidada import cargar_corridas, id_tarea, lista_combinaciones, particion
from bm.datos import cv_externa, cv_interna
from bm.espacios import ESPACIOS, JUSTIFICACION, REJILLAS, n_puntos_rejilla

assert SEED == cfg.SEED
RECALCULAR = False  # True: vuelve a ejecutar la validación anidada (cerca de una hora con 10 núcleos)
if RECALCULAR:
    subprocess.run([sys.executable, "scripts/ejecutar_experimentos.py"], check=True)
    subprocess.run([sys.executable, "scripts/ejecutar_finales.py", "--computacional"], check=True)

RES = cfg.RUTA_RESULTADOS
NM, NB, NO = cfg.NOMBRE_MODELO, cfg.NOMBRE_BALANCEO, cfg.NOMBRE_OPTIMIZADOR
# Paleta categórica validada, en orden fijo (la identidad sigue a la entidad, nunca al rango)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_MODELO = {m: CAT[i] for i, m in enumerate(cfg.MODELOS_CLF)}
COLOR_MODELO_REG = {m: CAT[i] for i, m in enumerate(cfg.MODELOS_REG)}
ORDEN_OPT = ["grid", "random", "bayes", "genetico", "halving"]
COLOR_OPT = {o: CAT[i] for i, o in enumerate(ORDEN_OPT)}
COLOR_BAL = {b: CAT[i] for i, b in enumerate(cfg.BALANCEOS)}

def ms(serie, d=3):
    """Media ± desviación estándar con coma decimal."""
    return f"{num(np.mean(serie), d)} ± {num(np.std(serie, ddof=1), d)}"


def etiqueta(r):
    """Nombre legible de una combinación."""
    partes = [NM[r["modelo"]]]
    if r["tarea"] == "clf":
        partes.append(NB[r["balanceo"]])
    partes.append(NO[r["optimizador"]])
    return " · ".join(partes)


versiones_e3 = pd.DataFrame({
    "Componente": ["imbalanced-learn", "xgboost", "optuna", "deap", "shap", "lime", "arch", "faiss-cpu",
                   "scikit-posthocs"],
    "Versión": [imblearn.__version__, xgboost.__version__, optuna.__version__, deap.__version__, shap.__version__,
                getattr(lime, "__version__", "0.2.0.1"), arch.__version__, faiss.__version__, sp.__version__]})
tabla(versiones_e3, indice=False)

DASH = Path("dashboard") / "datos"
DASH.mkdir(parents=True, exist_ok=True)


def exportar(nombre, obj):
    """Guarda un resultado para el dashboard y el Jupyter Book (mismas cifras que este cuaderno)."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        df_ = obj.copy()
        df_.columns = [" · ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df_.columns]
        for c in df_.columns:
            if df_[c].dtype == object:
                df_[c] = df_[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
        df_.to_parquet(DASH / f"{nombre}.parquet")
    else:
        (DASH / f"{nombre}.json").write_text(json.dumps(obj, default=float, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
'''

CARGA_RESULTADOS = r'''part = particion()
assert part.X_tr.index.equals(X_tr.index) and part.X_te.index.equals(X_te.index)  # misma partición del EDA

pliegues = pd.read_parquet(RES / "pliegues.parquet")
historial = pd.read_parquet(RES / "historial.parquet")
oof = pd.read_parquet(RES / "oof.parquet")
maestra = pd.read_parquet(RES / "tabla_maestra.parquet")
corridas = cargar_corridas()
finales = {}
for ruta in sorted((RES / "finales").glob("*_s42.pkl")) + [RES / "finales" / "clf_referencia_e2.pkl"]:
    with open(ruta, "rb") as f:
        r = pickle.load(f)
    finales[f"{r['tarea']}_{r['modelo']}"] = r


print(f"Corridas leídas: {len(corridas)} · combinaciones en la tabla maestra: {len(maestra)} · "
      f"modelos finales: {len(finales)}")'''

S38_MD = r'''### 3.8 Nuevos hallazgos para el tercer entregable

El desarrollo de este entregable exige dos análisis que el EDA anterior no cubría: la variable continua de la rama de regresión, que la guía pide aunque *Bank Marketing* sea un problema de clasificación, y el desbalance dentro de los pliegues de la validación anidada, que condiciona la viabilidad de SMOTE y ADASYN. Ambos se calculan solo con el conjunto de entrenamiento.

#### 3.8.1 Variable objetivo de la rama de regresión: `log1p(duration)`

La única variable continua del archivo con sentido como objetivo a nivel de cliente es `duration`, la duración de la llamada en segundos. En la clasificación se excluye porque solo se conoce después de la llamada (sección 3.5), pero como **objetivo** es legítima: estimar antes de marcar cuánto durará la llamada sirve para planificar la carga de los asesores. Las predictoras son las mismas de la clasificación (información previa a la llamada) y `y` no se usa, porque también se conoce después.'''

S381_CODE = r'''dur = tr["duration"].astype(float)
log_dur = np.log1p(dur)
resumen_dur = pd.DataFrame({
    "duration (s)": [dur.mean(), dur.median(), dur.std(), stats.skew(dur), (dur == 0).sum(), dur.max()],
    "log1p(duration)": [log_dur.mean(), log_dur.median(), log_dur.std(), stats.skew(log_dur), np.nan, log_dur.max()]},
    index=["Media", "Mediana", "Desv. est.", "Asimetría", "Registros con 0 s", "Máximo"])
tabla(resumen_dur, dec=2)

# Asociación con las predictoras previas a la llamada: Spearman (numéricas) y eta² (categóricas)
asoc = []
for c in VARS_NUM:
    rho, p_ = stats.spearmanr(tr[c], log_dur)
    asoc.append({"Variable": c, "Medida": "Spearman |rho|", "Valor": abs(rho), "p": p_})
for c in VARS_CAT:
    g = log_dur.groupby(tr[c])
    eta2 = (g.size() * (g.mean() - log_dur.mean()) ** 2).sum() / ((log_dur - log_dur.mean()) ** 2).sum()
    asoc.append({"Variable": c, "Medida": "eta²", "Valor": eta2, "p": stats.kruskal(*[v for _, v in g]).pvalue})
asoc = pd.DataFrame(asoc).sort_values("Valor", ascending=False)
tabla(asoc.head(8), dec=4, indice=False)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].hist(dur.clip(upper=dur.quantile(0.99)), bins=60, color=AZUL)
axs[0].set(title="duration (s), recortada en el percentil 99", xlabel="segundos", ylabel="registros")
axs[1].hist(log_dur, bins=60, color=AZUL)
axs[1].set(title="log1p(duration)", xlabel="log(1 + segundos)")
med_mes = log_dur.groupby(periodo_tr).median()
axs[2].plot(range(len(med_mes)), med_mes.values, color=AZUL, marker="o", ms=4)
axs[2].set_xticks(range(len(med_mes)), med_mes.index, rotation=60, ha="right", fontsize=7)
axs[2].set(title="Mediana mensual de log1p(duration)")
plt.tight_layout()
plt.show()

asim_log = stats.skew(log_dur)
top = asoc.iloc[0]
interpretar(
    f"La duración tiene media de {num(dur.mean())} s y mediana de {num(dur.median())} s, con asimetría de "
    f"{num(stats.skew(dur), 1)} y {int((dur == 0).sum())} llamadas de 0 s; en escala `log1p` la asimetría baja a "
    f"{num(asim_log, 2)}. La asociación más fuerte con una predictora previa a la llamada es la de `{top['Variable']}` "
    f"({top['Medida']} = {num(top['Valor'], 3)}); ninguna otra supera {num(asoc['Valor'].iloc[1], 3)}. La mediana mensual "
    f"se mueve entre {num(med_mes.min(), 2)} y {num(med_mes.max(), 2)} en escala log.",
    conectar("Esto significa que", "La transformación logarítmica vuelve la distribución casi simétrica y evita que unas "
             "pocas llamadas largas dominen el RMSE, pero la señal disponible antes de la llamada es débil: las "
             "asociaciones son estadísticamente significativas por el tamaño de muestra, aunque de magnitud pequeña. Es "
             "de esperar que los modelos de regresión apenas superen a la media histórica."),
    conectar("Por tal motivo, se opta por", "Modelar `log1p(duration)` y reportar el RMSE y el MAE en escala log y en "
             "segundos, comparando siempre contra el pronóstico de la media del entrenamiento. Un resultado nulo en esta "
             "rama no es un error del pipeline: es evidencia de que la duración depende de lo que ocurre durante la "
             "llamada."))'''

S382_MD = r'''#### 3.8.2 Desbalance dentro de los pliegues de la validación anidada

SMOTE necesita al menos `k + 1 = 6` positivos en cada ajuste para interpolar entre vecinos, y ADASYN además necesita positivos rodeados de negativos. Como la prevalencia de 2008 es muy baja y los pliegues internos de la validación anidada son pequeños, se verifica cuántos positivos tiene el ajuste más pequeño de cada pliegue externo antes de comprometer el diseño.'''

S382_CODE = r'''filas = []
for k_, (i_e, i_v) in enumerate(cv_externa(GAP).split(X_tr), start=1):
    internos = list(cv_interna(GAP).split(i_e))
    i_min = i_e[internos[0][0]]
    filas.append({"Pliegue externo": k_, "Positivos en el ajuste externo": int(y_tr.iloc[i_e].sum()),
                  "% positivos ajuste externo": 100 * y_tr.iloc[i_e].mean(),
                  "Filas del ajuste interno más pequeño": len(i_min),
                  "Positivos en el ajuste interno más pequeño": int(y_tr.iloc[i_min].sum()),
                  "Razón negativos/positivos (ajuste externo)": (1 - y_tr.iloc[i_e].mean()) / y_tr.iloc[i_e].mean(),
                  "% positivos validación externa": 100 * y_tr.iloc[i_v].mean()})
desb = pd.DataFrame(filas).set_index("Pliegue externo")
tabla(desb, dec=2)

interpretar(
    f"El ajuste interno más pequeño tiene {num(desb['Positivos en el ajuste interno más pequeño'].min())} positivos, "
    + ("por encima de los seis que necesita SMOTE con k = 5. " if desb['Positivos en el ajuste interno más pequeño'].min() >= 6
       else "por debajo de los seis que necesita SMOTE con k = 5. ") +
    f"La razón negativos/positivos del ajuste externo va de "
    f"{num(desb['Razón negativos/positivos (ajuste externo)'].min(), 1)} a "
    f"{num(desb['Razón negativos/positivos (ajuste externo)'].max(), 1)}, mientras que la prevalencia de validación "
    f"externa va del {num(desb['% positivos validación externa'].min(), 1)} % al "
    f"{num(desb['% positivos validación externa'].max(), 1)} %; en prueba es del {pct(y_te.mean())}.",
    conectar("Lo anterior indica que", "El sobremuestreo es viable en todos los pliegues, pero el desbalance que corrige "
             "no es el único problema: la prevalencia cambia entre periodos (*prior shift*), y ninguna técnica de "
             "balanceo, que equilibra las clases dentro del ajuste, anticipa ese cambio. Además, balancear desplaza las "
             "probabilidades hacia 0,5, de modo que su efecto debe medirse en métricas de umbral (recall, F1) y de "
             "calibración, no solo en AUC o AUC-PR, que dependen del orden."),
    conectar("Por tanto, corresponde", "Mantener SMOTE y ADASYN dentro del `Pipeline` de `imblearn` (solo actúan en `fit`), "
             "comparar el balanceo con métricas de umbral y de calibración (secciones 7.2 y 7.4) y separar su efecto del "
             "cambio de prevalencia, que se trata con la recalibración mensual heredada del Entregable 2."))'''
