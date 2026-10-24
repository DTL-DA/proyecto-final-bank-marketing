"""Sección 0: encabezado, configuración del entorno, carga de resultados y resumen ejecutivo."""

ENCABEZADO = r'''# Proyecto Final · Cuarto entregable · Modelo original para predecir la suscripción a depósitos a término (*Bank Marketing*, UCI)

**Logística de estado latente en dos escalas (L2E): diseño, implementación algorítmica y validación estadística frente a los modelos del tercer entregable**

| | |
|---|---|
| **Autor** | Donny Torres |
| **Programa** | Maestría en Analítica de Datos, Universidad del Norte (Barranquilla) |
| **Asignatura** | Machine Learning |
| **Fecha** | Octubre de 2026 |
| **Conjunto de datos** | *Bank Marketing* con contexto social y económico (`bank-additional-full.csv`) y tres conjuntos adicionales de OpenML |
| **Fuente** | UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/222/bank+marketing> (licencia CC BY 4.0) |
| **Tarea** | Clasificación de `y` (suscribe o no) con información previa a la llamada y validación cronológica |
| **Guía** | *Proyecto Final: Dashboard Predictivo, Modelo Original e Informe Reproducible* |

## 0. Introducción

Este cuaderno desarrolla el componente de investigación del proyecto: **un modelo propio**, que no proviene de un `import`, se implementa algorítmicamente y se compara estadísticamente contra los modelos del tercer entregable. Mantiene las mismas condiciones de ese entregable, esto es, el mismo conjunto de datos, la misma partición cronológica 80/20, el mismo preprocesamiento, los mismos pliegues externos e internos de la validación anidada, la misma semilla y el mismo presupuesto de optimización, de modo que cualquier diferencia se debe al modelo y no al protocolo.

**Pregunta de investigación.** ¿Un modelo que sigue día a día la tasa de suscripción puede predecir, antes de la llamada, qué clientes suscribirán un depósito a término mejor que los clasificadores estáticos?

Retoma la pregunta del Entregable 1 (sección 1.1), que medía la capacidad de discriminación de una regresión logística con información previa a la llamada y su estabilidad en el orden temporal, y la lleva al modelo original. Los clasificadores estáticos son los siete modelos del tercer entregable (logística, Naive Bayes, KNN, árbol de decisión, Random Forest, XGBoost y SVM), que asumen que la relación entre el cliente y la suscripción no cambia en el tiempo.

**Productos en línea.**

- **Dashboard predictivo:** <https://proyecto-final-bank-marketing.onrender.com> (páginas Contexto, EDA y Modelos, con el L2E integrado; en el plan gratuito de Render la primera carga tras 15 minutos sin visitas tarda cerca de un minuto).
- **Jupyter Book:** <https://dtl-da.github.io/proyecto-final-bank-marketing/cuarto-entregable/>.
- **Repositorio:** <https://github.com/DTL-DA/proyecto-final-bank-marketing>; este cuaderno, su código y sus resultados están en la carpeta `cuarto_entregable/`.

El punto de partida es una evaluación crítica de lo que ya se hizo. Las 140 combinaciones del tercer entregable no se distinguen entre sí, dentro de 2008 ninguna discrimina mejor que el azar y la mejor mejora en la prueba no vino de un modelo más flexible sino de recalibrar el intercepto cada mes. Ese patrón tiene una explicación: la tasa de suscripción de cada día es un **nivel latente persistente** que los modelos estáticos no pueden seguir. Sobre ese diagnóstico se propone el modelo L2E, una regresión logística cuyo intercepto es un estado latente diario que se mueve en una escala rápida y cuyas pendientes derivan en una escala lenta, que se ajusta por máxima verosimilitud penalizada con un algoritmo de Newton por bloques y que, en producción, se actualiza con un filtro de Laplace a medida que se conocen las etiquetas de los días ya cerrados.

**Organización.** La numeración de las secciones sigue los capítulos del Jupyter Book que pide la guía (cuadro 4), así que cada sección del cuaderno es un capítulo del libro:

| Sección | Contenido | Bloque o sección de la guía |
|---|---|---|
| 0. Introducción | Resumen ejecutivo, configuración y carga de resultados | 9.1 |
| 1. Trazabilidad de correcciones | Evaluación crítica del tercer entregable y tabla de correcciones | Bloque A |
| 2. Datos: ETL y EDA | Partición heredada y nuevo hallazgo: el nivel diario latente | 9.1 |
| 3. Modelos benchmark | Mejores combinaciones por modelo y mejor combinación global | Bloque B |
| 4. Revisión de literatura y novedad | Protocolo, matriz de novedad, trabajos cercanos y declaración | 5.2 |
| 5. Modelo original | Formulación, pseudocódigo, propiedades, código, pruebas unitarias, complejidad, optimización y ablación | Bloque C, 5.3 y 5.4 |
| 6. Validación estadística | Protocolo, pruebas de hipótesis, controles, robustez, conjuntos adicionales e interpretabilidad | Bloque D y sección 6 |
| 7. Dashboard y despliegue | Artefacto del modelo original, prueba de ida y vuelta, predicción en vivo, arquitectura, decisiones de diseño, guía de despliegue, capturas y URL pública | Bloques B, E y F |
| 8. Conclusiones y limitaciones | Hallazgos, veredicto sobre R1, R2 y R3, limitaciones, trabajo futuro y uso de IA | 9.1 y 13 |
| 9. Reproducibilidad y bibliografía | Entorno, semillas, tabla maestra de experimentos y referencias | 9.1 y 10 |

**Ejecución.** La validación anidada del modelo original (7 variantes × 5 pliegues, con Optuna en el bucle interno), los modelos finales, los tres conjuntos adicionales, los análisis de sensibilidad y la revisión de literatura se ejecutan con los *scripts* de `scripts/` y quedan registrados en `resultados/`; este cuaderno los lee y calcula aquí todas las pruebas estadísticas. Con `RECALCULAR = True` en la celda de configuración se vuelven a lanzar los *scripts*. Todas las interpretaciones se generan con los valores calculados en la misma ejecución, de forma que las cifras del texto coinciden siempre con las salidas.'''

CONFIG_MD = r'''### 0.1 Configuración del entorno y reproducibilidad

La semilla global es la misma del tercer entregable (`SEED = 42`, en `src/bm/config.py`) y se propaga a `numpy`, scikit-learn, SMOTE y ADASYN, XGBoost, Optuna y DEAP. El modelo L2E es determinista: su ajuste maximiza un objetivo estrictamente cóncavo con Newton, así que la semilla solo afecta al optimizador de hiperparámetros, cuya estabilidad se mide con tres semillas adicionales (sección 6.9). El entorno es `bm_e3` (Python 3.12) y las versiones exactas están en `requirements.txt`.'''

CONFIG_CODE = r'''import json
import pickle
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import Markdown, display
from scipy import stats
from scipy.special import expit
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*")
sys.path.insert(0, str(Path("src").resolve()))

import optuna
import scikit_posthocs as sp

from bm import config as cfg
from bm import estadistica as st
from bm import estadistica_e4 as st4
from bm import evaluacion as evm
from bm.adicionales import DESCRIPCION as DESC_ADIC
from bm.anidada import particion
from bm.datos import cv_externa, cv_interna
from bm.evaluacion import elegir_finalistas
from bm.modelo_original import LogisticaDosEscalas, filtrar_nivel
from bm.protocolo_original import (NOMBRE_VARIANTE, VARIANTES, con_dia, construir_pipeline_original, em_por_dia,
                                   logit_seguro, puntaje_secuencial)

SEED = cfg.SEED
np.random.seed(SEED)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)
RECALCULAR = False  # True: vuelve a ejecutar todos los scripts de resultados/ (cerca de 45 minutos con 10 núcleos)
if RECALCULAR:
    for script in ("importar_e3.py", "ejecutar_original.py", "ejecutar_adicionales.py", "ejecutar_analisis.py",
                   "revision_literatura.py", "verificar_referencias.py"):
        subprocess.run([sys.executable, f"scripts/{script}"], check=True)

# Paleta validada del proyecto: categórica en orden fijo, tinta y rejilla recesivas
AZUL, NARANJA, AGUAMARINA, AMARILLO, ROJO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
TINTA, TINTA_2, GRIS, REJILLA, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("secuencial", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list("divergente", [AZUL, "#f0efec", ROJO])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "figure.titlesize": 12, "figure.titleweight": "bold",
    "axes.prop_cycle": mpl.cycler(color=[AZUL, NARANJA, AGUAMARINA, AMARILLO]), "lines.linewidth": 2,
})


def num(x, d=0):
    """Número con coma decimal y espacio de miles."""
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def num_c(x):
    """Valor de un hiperparámetro sin ceros sobrantes (0,001; 0,1; 100)."""
    return f"{x:.3g}".replace(".", ",")


def pct(x, d=1):
    return f"{num(100 * x, d)} %"


def pval(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def p_txt(p):
    return "p < 0,001" if p < 0.001 else f"p = {num(p, 3)}"


def p_frase(p):
    """Valor p para insertar en una frase: «menor que 0,001» o «de 0,123»."""
    return "menor que 0,001" if p < 0.001 else f"de {num(p, 3)}"


def interpretar(interpretacion, implicacion, accion):
    display(Markdown(f"**Interpretación.** {interpretacion}\n\n{implicacion}\n\n{accion}"))


def conectar(conector, texto):
    """Antepone un conector al texto y pasa a minúscula su primera letra."""
    return f"{conector} {texto[:1].lower()}{texto[1:]}" if texto else texto


def tabla(df, dec=3, indice=True):
    sty = df.style.format(precision=dec, decimal=",", thousands=" ", na_rep="—")
    if not indice:
        sty = sty.hide(axis="index")
    display(sty)


def enumerar(items, codigo=False):
    """Une una lista en español: a, b y c."""
    items = [f"`{i}`" if codigo else str(i) for i in items]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def para_guion(nombre, obj):
    """Guarda una tabla o un diccionario para el guion de sustentación (mismas cifras que el cuaderno)."""
    destino = Path("resultados") / "guion"
    destino.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(destino / f"{nombre}.parquet")
    else:
        (destino / f"{nombre}.json").write_text(json.dumps(obj, default=float, ensure_ascii=False, indent=1),
                                               encoding="utf-8")


def ms(serie, d=3):
    """Media ± desviación estándar con coma decimal."""
    return f"{num(np.mean(serie), d)} ± {num(np.std(serie, ddof=1), d)}"


versiones = pd.DataFrame({
    "Componente": ["Python", "numpy", "pandas", "scipy", "scikit-learn", "optuna", "scikit-posthocs", "matplotlib"],
    "Versión": [platform.python_version(), np.__version__, pd.__version__, scipy.__version__, sklearn.__version__,
                optuna.__version__, sp.__version__, mpl.__version__]})
tabla(versiones, indice=False)
print(f"Semilla global: SEED = {SEED}")'''

CARGA_MD = r'''### 0.2 Carga de la partición y de los resultados

Se reconstruye la partición cronológica con el mismo código del tercer entregable (`bm.anidada.particion`) y se leen tres grupos de resultados: los del tercer entregable (`resultados/e3`, importados con `scripts/importar_e3.py`), los del modelo original (validación anidada, ablación, semillas y modelos finales) y los complementarios (conjuntos adicionales, sensibilidad, SHAP, complejidad y revisión de literatura).'''

CARGA_CODE = r'''part = particion()
X_tr, X_te, y_tr, y_te = part.X_tr, part.X_te, part.y_tr, part.y_te
tiempo_tr, tiempo_te, GAP = part.tiempo_tr, part.tiempo_te, part.gap
periodo_tr, periodo_te = tiempo_tr["periodo"], tiempo_te["periodo"]
dia_tr, dia_te = tiempo_tr["dia"].to_numpy(), tiempo_te["dia"].to_numpy()
tiempo = pd.concat([tiempo_tr, tiempo_te])
yte = y_te.to_numpy()

RES = cfg.RUTA_RESULTADOS
E3 = RES / "e3"
maestra = pd.read_parquet(E3 / "tabla_maestra.parquet")
pliegues_e3 = pd.read_parquet(E3 / "pliegues.parquet")
oof_e3 = pd.read_parquet(E3 / "oof_finalistas.parquet")
with open(E3 / "finales_clf.pkl", "rb") as f:
    fin_e3 = pickle.load(f)
fin_ext = elegir_finalistas(maestra, "clf")
MODELOS = fin_ext["modelo"].tolist()  # ordenados por AUC-PR externo medio (el primero es el mejor global)
MEJOR = MODELOS[0]


def leer(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


corr = {v: [leer(RES / "corridas_original" / f"{v}_{k}.pkl") for k in range(cfg.K_EXTERNO)] for v in VARIANTES}
corr_sem = {s: [leer(RES / "corridas_original" / f"l2e_{k}_s{s}.pkl") for k in range(cfg.K_EXTERNO)] for s in (1, 2, 3)}
fin_l2e = {v: leer(RES / "finales_original" / f"{v}.pkl") for v in VARIANTES}
fin_sem = {s: leer(RES / "finales_original" / f"l2e_s{s}.pkl") for s in (1, 2, 3)}
L2E = fin_l2e["l2e"]
adic = pd.read_parquet(RES / "adicionales.parquet")
ANA = RES / "analisis"
literatura = json.loads((RES / "literatura" / "consultas.json").read_text(encoding="utf-8"))

# Nombres y colores: la identidad sigue a la entidad, nunca al rango
NOMBRE = {"L2E": "L2E (modelo original)", "referencia_e2": "Referencia E2 (logística C = 0,01)"}
for r_ in fin_ext.itertuples():
    NOMBRE[r_.modelo] = f"{cfg.NOMBRE_MODELO[r_.modelo]} · {cfg.NOMBRE_BALANCEO[r_.balanceo]} · " \
                        f"{cfg.NOMBRE_OPTIMIZADOR[r_.optimizador]}"
CORTO = {"L2E": "L2E", "referencia_e2": "Referencia E2", **{m: cfg.NOMBRE_MODELO[m] for m in MODELOS}}
COLOR = {"L2E": TINTA, "referencia_e2": GRIS, **{m: CAT[cfg.MODELOS_CLF.index(m)] for m in MODELOS}}

# Puntajes de prueba de todos los modelos comparados (mismas 8 236 observaciones, mismo orden)
PUNTAJE_TE = {"L2E": L2E["puntaje_te"], **{m: fin_e3[f"clf_{m}"]["puntaje_te"] for m in MODELOS},
              "referencia_e2": fin_e3["clf_referencia_e2"]["puntaje_te"]}
ES_PROBA = {"L2E": True, **{m: fin_e3[f"clf_{m}"]["es_proba"] for m in MODELOS}, "referencia_e2": True}

# Predicciones fuera de muestra del bucle externo (mismos pliegues para todos)
oof_l2e = pd.concat([pd.DataFrame({"modelo": "L2E", "pliegue": c["pliegue"], "indice": c["oof"]["indice"],
                                   "y": c["oof"]["y"], "puntaje": c["oof"]["puntaje"]}) for c in corr["l2e"]])
oof = pd.concat([oof_e3[["modelo", "pliegue", "indice", "y", "puntaje"]].astype({"modelo": str}), oof_l2e],
                ignore_index=True)
oof["periodo"] = tiempo.loc[oof["indice"], "periodo"].to_numpy()
oof["dia"] = tiempo.loc[oof["indice"], "dia"].to_numpy()
assert oof.groupby("modelo")["indice"].apply(lambda s: tuple(np.sort(s))).nunique() == 1  # mismas observaciones

print(f"Partición: {num(len(X_tr))} registros de entrenamiento y {num(len(X_te))} de prueba · "
      f"{len(maestra)} combinaciones del tercer entregable · {sum(len(v) for v in corr.values())} corridas anidadas "
      f"del modelo original ({len(VARIANTES)} variantes) · {len(adic)} corridas en conjuntos adicionales")'''

RESUMEN_CODE = r'''ext_l2e = np.array([c["auc_pr"] for c in corr["l2e"]])
ext_mejor = pliegues_e3.merge(fin_ext.iloc[[0]][["modelo", "balanceo", "optimizador"]])["auc_pr"].to_numpy()
ap_te = {k: average_precision_score(yte, s) for k, s in PUNTAJE_TE.items()}
auc_te = {k: roc_auc_score(yte, s) for k, s in PUNTAJE_TE.items()}
mejor_te = max((k for k in ap_te if k != "L2E"), key=ap_te.get)
display(Markdown(f"""### Resumen ejecutivo

- **Problema.** Predecir, antes de la llamada, qué clientes suscribirán un depósito a término, en un periodo en que la tasa de suscripción pasa del {pct(y_tr.mean())} (entrenamiento, 2008-2009) al {pct(y_te.mean())} (prueba, 2009-2010).
- **Pregunta de investigación.** ¿Un modelo que sigue día a día la tasa de suscripción puede predecir, antes de la llamada, qué clientes suscribirán un depósito a término mejor que los clasificadores estáticos?
- **Diagnóstico del tercer entregable.** La mejor de las 140 combinaciones, {NOMBRE[MEJOR]}, logra un AUC-PR externo de {ms(ext_mejor)}; en los pliegues de 2008 ningún modelo supera de forma apreciable a la prevalencia, y en la prueba el mejor AUC-PR es el de {CORTO[mejor_te]} ({num(ap_te[mejor_te], 3)}). La señal que falta es temporal: la tasa diaria de suscripción es un nivel latente persistente (sección 2.2).
- **Modelo original.** L2E: regresión logística con un nivel diario latente en paseo aleatorio (escala rápida), pendientes que derivan lentamente (escala lenta), ajuste por moda posterior con Newton por bloques en O(T) y filtro de Laplace con etiquetas diferidas un día. Implementado desde cero con `numpy` y `scipy` (`src/bm/modelo_original.py`), con {len([l for l in Path('tests/test_modelo_original.py').read_text(encoding='utf-8').splitlines() if l.startswith('def test_')])} funciones de prueba unitaria.
- **Resultado.** En el bucle externo, L2E obtiene {ms(ext_l2e)} frente a {ms(ext_mejor)} de la mejor combinación benchmark; en la prueba, AUC-PR de {num(ap_te['L2E'], 3)} y AUC de {num(auc_te['L2E'], 3)}, frente a {num(ap_te[mejor_te], 3)} y {num(max(v for k, v in auc_te.items() if k != 'L2E'), 3)} del mejor benchmark en cada métrica. La sección 6 verifica si la diferencia es significativa, con qué pruebas y de qué magnitud, y dónde el modelo no supera a los benchmarks."""))'''
