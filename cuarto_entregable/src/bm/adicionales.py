"""Conjuntos de datos adicionales para la evaluación amplia del modelo original (guía, cuadro 1).

Tres problemas de clasificación binaria con orden temporal y de naturaleza distinta a Bank Marketing, todos públicos en
OpenML. Cada uno se evalúa con el mismo protocolo: validación anidada cronológica (``TimeSeriesSplit`` de 5 pliegues
externos y 3 internos, con separación de un día), AUC-PR como métrica, 20 configuraciones por optimizador y, para
cada modelo base, la combinación de balanceo y optimizador que ganó en Bank Marketing.
"""

import pickle
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .anidada import metricas_clf, puntajes
from .modelos import _estimador_clf
from .optimizadores import OPTIMIZADORES, Evaluador
from .protocolo_original import ap_secuencial, clave_espacio, construir_pipeline_original, puntaje_secuencial

RUTA_DATOS_ADICIONALES = cfg.RAIZ / "data" / "adicionales"
RUTA_CORRIDAS_ADICIONALES = cfg.RUTA_RESULTADOS / "corridas_adicionales"
POR_DIA_AIRLINES = 700
DIAS_ELECTRICITY = 450

DESCRIPCION = {
    "airlines": {"openml": 1169, "dominio": "Transporte aéreo", "objetivo": "el vuelo se retrasa (Delay = 1)",
                 "periodo": "un mes de vuelos comerciales en EE. UU. (31 días)",
                 "fuente": "Ikonomovska (2009), Data Expo de la ASA; OpenML 1169"},
    "electricity": {"openml": 151, "dominio": "Mercado eléctrico",
                    "objetivo": "el precio de Nueva Gales del Sur sube frente a la media de 24 h (UP)",
                    "periodo": "últimos 450 días de 1996-1998, 48 periodos de media hora por día",
                    "fuente": "Harries (1999); OpenML 151"},
    "ozone": {"openml": 1487, "dominio": "Calidad del aire", "objetivo": "día con ozono alto (umbral de 8 horas)",
              "periodo": "1998-2004, un registro por día", "fuente": "Zhang y Fan (2008), UCI; OpenML 1487"},
}


@dataclass
class Conjunto:
    """Conjunto adicional listo para la validación anidada."""
    nombre: str
    X: pd.DataFrame
    y: pd.Series
    numericas: list
    categoricas: list
    gap: int


def cargar(nombre: str, seed: int = cfg.SEED) -> Conjunto:
    """Descarga (o lee de la caché local) y prepara un conjunto adicional, en orden cronológico.

    - ``airlines``: el día se reconstruye con los cambios de ``DayOfWeek`` (31 días); se toman ``POR_DIA_AIRLINES``
      vuelos al azar por día, en su orden original, y se descarta ``Flight`` (identificador).
    - ``electricity``: el día se define por el reinicio de ``period`` (48 periodos por día) y se usan los últimos
      ``DIAS_ELECTRICITY`` días, cuando las variables de Victoria ya no están imputadas con una constante.
    - ``ozone``: un registro por día; los faltantes se imputan con la mediana dentro del pipeline.
    """
    info = DESCRIPCION[nombre]
    b = fetch_openml(data_id=info["openml"], data_home=RUTA_DATOS_ADICIONALES, as_frame=True, parser="auto")
    X, objetivo = b.data.copy(), b.target.astype(str)
    if nombre == "airlines":
        dow = X["DayOfWeek"].astype(int).to_numpy()
        X["dia"] = np.r_[0, np.cumsum(np.diff(dow) != 0)]
        y = (objetivo == "1").astype(int)
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(X["dia"].to_numpy() == d), POR_DIA_AIRLINES,
                                                 replace=False) for d in np.unique(X["dia"])]))
        X, y = X.iloc[idx].drop(columns=["Flight"]), y.iloc[idx]
        num, cat = ["Time", "Length"], ["Airline", "AirportFrom", "AirportTo", "DayOfWeek"]
    elif nombre == "electricity":
        per = X["period"].astype(float).to_numpy()
        X["dia"] = np.r_[0, np.cumsum(np.diff(per) < 0)]
        y = (objetivo == "UP").astype(int)
        m = X["dia"] >= X["dia"].max() + 1 - DIAS_ELECTRICITY
        X, y = X.loc[m].drop(columns=["date"]), y.loc[m]
        num, cat = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"], ["day"]
    elif nombre == "ozone":
        y = (objetivo == "2").astype(int)
        X["dia"] = np.arange(len(X))
        num, cat = [c for c in X.columns if c != "dia"], []
    else:
        raise ValueError(f"Conjunto desconocido: {nombre}")
    for c in cat:
        X[c] = X[c].astype(str)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True).rename("y")
    gap = int(np.ceil(X["dia"].value_counts().quantile(0.95)))
    return Conjunto(nombre, X, y, num, cat, gap)


def preprocesador(conj: Conjunto, con_tiempo: bool = False) -> ColumnTransformer:
    """Imputación por mediana y estandarización de numéricas; *one-hot* de categóricas (mismas reglas del E3)."""
    pasos = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), conj.numericas)]
    if conj.categoricas:
        pasos.append(("cat", OneHotEncoder(drop="first", min_frequency=cfg.MIN_FREQ,
                                           handle_unknown="infrequent_if_exist", sparse_output=False),
                      conj.categoricas))
    if con_tiempo:
        pasos.append(("tiempo", "passthrough", ["dia"]))
    return ColumnTransformer(pasos)


def pipeline_base(conj: Conjunto, modelo: str, balanceo: str, seed: int = cfg.SEED) -> Pipeline:
    """Pipeline preprocesamiento → (balanceo) → modelo base, como ``modelos.construir_pipeline``."""
    pasos = [("pre", preprocesador(conj))]
    if balanceo == "smote":
        pasos.append(("bal", SMOTE(random_state=seed)))
    elif balanceo == "adasyn":
        pasos.append(("bal", ADASYN(random_state=seed)))
    return Pipeline(pasos + _estimador_clf(modelo, balanceo == "class_weight", seed))


def ejecutar_adicional(conj: Conjunto, modelo: str, balanceo: str, optimizador: str, pliegue: int,
                       variante: str = "l2e", seed: int = cfg.SEED, guardar: bool = True) -> dict:
    """Un modelo en un pliegue externo de un conjunto adicional (``modelo == "original"`` para el L2E)."""
    etiqueta = variante if modelo == "original" else f"{modelo}_{balanceo}_{optimizador}"
    ruta = RUTA_CORRIDAS_ADICIONALES / f"{conj.nombre}_{etiqueta}_{pliegue}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    i_ent, i_val = list(TimeSeriesSplit(n_splits=cfg.K_EXTERNO, gap=conj.gap).split(conj.X))[pliegue]
    X_ent, y_ent, X_val, y_val = conj.X.iloc[i_ent], conj.y.iloc[i_ent], conj.X.iloc[i_val], conj.y.iloc[i_val]
    cv_int = TimeSeriesSplit(n_splits=cfg.K_INTERNO, gap=conj.gap)
    original = modelo == "original"
    if original:
        clave = clave_espacio(variante)
        fabrica = lambda: construir_pipeline_original(variante, preprocesador(conj, con_tiempo=True))  # noqa: E731
        ev = Evaluador(fabrica(), X_ent, y_ent, cv_int, ap_secuencial)
    else:
        clave = modelo
        fabrica = lambda: pipeline_base(conj, modelo, balanceo, seed)  # noqa: E731
        ev = Evaluador(fabrica(), X_ent, y_ent, cv_int, cfg.METRICA_CLF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        OPTIMIZADORES["bayes" if original else optimizador](ev, "clf", clave, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = fabrica().set_params(**params)
        try:
            t = time.perf_counter()
            final.fit(X_ent, y_ent)
            t_fit = time.perf_counter() - t
            if original:
                s, es_proba = puntaje_secuencial(final, X_val, y_val), True
                pred = (s >= 0.5).astype(int)
            else:
                s, es_proba = puntajes(final, X_val)
                pred = final.predict(X_val)
            met = metricas_clf(y_val, s, pred, es_proba)
        except ValueError as error:  # p. ej., SMOTE sin vecinos suficientes en un ajuste muy pequeño
            t_fit, s, met = np.nan, np.full(len(y_val), np.nan), {"auc_pr": np.nan, "auc": np.nan,
                                                                  "error": str(error)}
    res = {"conjunto": conj.nombre, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
           "variante": variante if original else None, "pliegue": pliegue, "seed": seed, "params": params,
           "puntaje_interno": interno, "t_opt": t_opt, "t_fit": t_fit, "n_ent": len(i_ent), "n_val": len(i_val),
           "prevalencia_val": float(y_val.mean()), **met,
           "oof": {"indice": X_val.index.to_numpy(), "y": y_val.to_numpy(), "puntaje": np.asarray(s, dtype=float),
                   "dia": X_val["dia"].to_numpy()}}
    if guardar:
        RUTA_CORRIDAS_ADICIONALES.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def resumen_conjunto(conj: Conjunto) -> dict:
    """Tamaño, prevalencia, días y autocorrelación del logit de la tasa diaria (estructura que explota el L2E)."""
    g = pd.DataFrame({"d": conj.X["dia"].to_numpy(), "y": conj.y.to_numpy()}).groupby("d")["y"].agg(["sum", "size"])
    lt = np.log((g["sum"] + 0.5) / (g["size"] - g["sum"] + 0.5))
    return {"conjunto": conj.nombre, "registros": len(conj.X), "predictoras": len(conj.numericas) + len(conj.categoricas),
            "dias": len(g), "registros_por_dia": float(g["size"].median()), "prevalencia": float(conj.y.mean()),
            "autocorr_1": float(lt.autocorr(1)) if len(g) > 2 else np.nan, "gap": conj.gap}


METRICAS_RESUMEN = ["auc_pr", "auc", "brier", "log_loss", "accuracy", "f1", "t_opt", "t_fit", "prevalencia_val"]


@lru_cache(maxsize=None)
def conjunto_cacheado(nombre: str) -> Conjunto:
    """Conjunto preparado, una vez por proceso (los procesos de ``joblib`` lo cargan por nombre)."""
    return cargar(nombre)


def tarea_adicional(nombre, modelo, balanceo, optimizador, pliegue, variante=None) -> dict:
    """Ejecuta una tarea por nombre de conjunto y devuelve su resumen (las predicciones quedan en disco)."""
    r = ejecutar_adicional(conjunto_cacheado(nombre), modelo, balanceo, optimizador, pliegue, variante or "l2e")
    return {"conjunto": nombre, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
            "variante": r["variante"], "pliegue": pliegue, **{m: r.get(m) for m in METRICAS_RESUMEN}}
