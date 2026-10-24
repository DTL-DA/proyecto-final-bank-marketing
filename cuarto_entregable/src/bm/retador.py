"""Modelo retador del L2E: XGBoost sobre el nivel reciente (XGB-N), con la misma información que el L2E.

El L2E combina en el logit un nivel diario, que actualiza con las etiquetas de los días ya cerrados, y un efecto
lineal de las variables del cliente. El retador conserva esa estructura y cambia la parte del cliente por *gradient
boosting*:

    logit P(y = 1) = a + b · logit(tasa reciente) + f(x),

donde la tasa reciente es un promedio exponencial de la tasa diaria de suscripción de los días anteriores, a y b se
estiman con una logística de una variable y f es un XGBoost ajustado sobre ese margen (``base_margin``). El nivel
entra de forma lineal en el logit y por eso extrapola, a diferencia de un árbol, cuyas hojas quedan fijas en el rango
de entrenamiento; las variables macroeconómicas y ``month`` se retiran porque el nivel cumple su papel y porque son las
que salen del rango en la prueba (sección 6.13 del cuaderno).

Ningún rasgo usa etiquetas del propio día ni de días posteriores (retardo de un día, igual que el L2E). En los
pliegues externos los rasgos se calculan sin los días de separación (``gap``), que el L2E tampoco asimila.

Variantes, todas con el protocolo del tercer entregable (validación anidada 5 × 3, Optuna con 20 configuraciones):

- ``xgb_n``: el retador descrito arriba.
- ``xgb_t``: XGBoost con los rezagos como rasgos de entrada (práctica estándar), sin macro ni ``month``.
- ``logit_t``: logística L2 con los mismos rasgos que ``xgb_t``.
"""

import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from . import config as cfg
from .anidada import metricas_clf, particion
from .datos import cv_externa, cv_interna
from .espacios import ESPACIOS, C
from .optimizadores import OPTIMIZADORES, Evaluador
from .preprocesamiento import construir_preprocesador

RUTA_RETADOR = cfg.RUTA_RESULTADOS / "retador"
SEMIVIDAS = (1, 3, 10, 30)
MACRO = ["cons.price.idx", "cons.conf.idx", "nr.employed"]
TASAS = ["tasa_ayer"] + [f"tasa_ewm_{h}" for h in SEMIVIDAS]
RASGOS = TASAS + ["log_volumen_ewm_3"]
VARIANTES_RETADOR = ("xgb_n", "xgb_t", "logit_t")


def rasgos_rezagados(dias, y) -> pd.DataFrame:
    """Rasgos por registro calculados solo con los días anteriores a su día de contacto.

    Parameters
    ----------
    dias : array-like
        Día de contacto de cada registro, en orden temporal.
    y : array-like
        Etiquetas de los mismos registros.
    """
    d = pd.DataFrame({"dia": np.asarray(dias), "y": np.asarray(y, dtype=float)})
    g = d.groupby("dia")["y"].agg(["sum", "size"])
    salida = pd.DataFrame(index=g.index)
    salida["tasa_ayer"] = (g["sum"] / g["size"]).shift(1)
    for h in SEMIVIDAS:
        a = 1 - 0.5 ** (1 / h)
        salida[f"tasa_ewm_{h}"] = (g["sum"].ewm(alpha=a).mean() / g["size"].ewm(alpha=a).mean()).shift(1)
    salida["log_volumen_ewm_3"] = np.log1p(g["size"].ewm(alpha=1 - 0.5 ** (1 / 3)).mean()).shift(1)
    # El primer día no tiene pasado: se usa la tasa del primer 1 % de los registros (solo entrenamiento)
    tasa_inicial = d["y"].iloc[: max(1, int(0.01 * len(d)))].mean()
    salida = salida.fillna({c: tasa_inicial for c in TASAS}).bfill()
    return salida.reindex(d["dia"].to_numpy()).set_index(d.index)


def con_rasgos(X, dias, y) -> pd.DataFrame:
    R = rasgos_rezagados(np.asarray(dias), np.asarray(y))
    R.index = X.index
    return X.join(R)


def _logit_tasa(X):
    return logit(np.clip(np.asarray(X, dtype=float), 1e-3, 1 - 1e-3))


def _preprocesador(variante):
    num = [c for c in cfg.VARS_NUM if c not in MACRO]
    cat = [c for c in cfg.VARS_CAT if c != "month"]
    base = construir_preprocesador(num, cat).transformers
    if variante == "xgb_n":
        return ColumnTransformer(base + [("nivel", "passthrough", RASGOS)])  # al final y sin transformar
    tasas = make_pipeline(FunctionTransformer(_logit_tasa, feature_names_out="one-to-one"), StandardScaler())
    return ColumnTransformer(base + [("tasas", tasas, TASAS), ("volumen", StandardScaler(), ["log_volumen_ewm_3"])])


class XGBNivel(ClassifierMixin, BaseEstimator):
    """XGBoost ajustado sobre el margen del nivel reciente: logit P = a + b · logit(tasa) + f(x).

    Espera que las últimas ``len(RASGOS)`` columnas de ``X`` sean los rasgos rezagados sin transformar; ``nivel``
    elige cuál de las tasas define el margen. Los árboles solo ven las variables del cliente.
    """

    def __init__(self, nivel="tasa_ewm_3", n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
                 random_state=cfg.SEED):
        self.nivel = nivel
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

    def _separar(self, X):
        X = np.asarray(X, dtype=float)
        k = len(RASGOS)
        ell = _logit_tasa(X[:, X.shape[1] - k + RASGOS.index(self.nivel)])
        return X[:, :-k], ell.reshape(-1, 1)

    def _margen(self, ell):
        return self.calibracion_.decision_function(ell)

    def fit(self, X, y):
        Xc, ell = self._separar(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.calibracion_ = LogisticRegression(C=1e4, max_iter=1000).fit(ell, y)
        self.arboles_ = XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      max_depth=self.max_depth, subsample=self.subsample, colsample_bytree=0.8,
                                      tree_method="hist", eval_metric="logloss", n_jobs=1,
                                      random_state=self.random_state, verbosity=0)
        self.arboles_.fit(Xc, y, base_margin=self._margen(ell))
        return self

    def decision_function(self, X):
        Xc, ell = self._separar(X)
        return self.arboles_.predict(Xc, base_margin=self._margen(ell), output_margin=True)

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def pipeline_retador(variante, seed=cfg.SEED):
    """Pipeline preprocesamiento (con rasgos rezagados) → modelo."""
    if variante == "xgb_n":
        modelo = XGBNivel(random_state=seed)
    elif variante == "xgb_t":
        modelo = XGBClassifier(tree_method="hist", colsample_bytree=0.8, eval_metric="logloss", n_jobs=1,
                               random_state=seed, verbosity=0)
    else:
        modelo = LogisticRegression(max_iter=5000, random_state=seed)
    return Pipeline([("pre", _preprocesador(variante)), ("modelo", modelo)])


ESPACIOS_RETADOR = {
    "xgb_n": {**ESPACIOS[("clf", "xgboost")], "modelo__nivel": C(*TASAS[1:])},
    "xgb_t": ESPACIOS[("clf", "xgboost")],
    "logit_t": {"modelo__C": ESPACIOS[("clf", "logistica")]["modelo__C"]},
}
for _v, _e in ESPACIOS_RETADOR.items():
    ESPACIOS[("clf", f"retador_{_v}")] = _e


def ejecutar_retador(variante, pliegue, seed=cfg.SEED, guardar=True):
    """Una variante del retador en un pliegue externo, o el modelo final evaluado en la prueba si ``pliegue == -1``."""
    sufijo = "prueba" if pliegue == -1 else str(pliegue)
    ruta = RUTA_RETADOR / f"{variante}_{sufijo}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    if pliegue == -1:
        # Prueba: sigue al entrenamiento sin separación, como en el L2E final
        X = con_rasgos(pd.concat([p.X_tr, p.X_te]), pd.concat([p.tiempo_tr, p.tiempo_te])["dia"],
                       pd.concat([p.y_tr, p.y_te]))
        X_ent, y_ent, X_val, y_val = X.loc[p.X_tr.index], p.y_tr, X.loc[p.X_te.index], p.y_te
        cv = TimeSeriesSplit(n_splits=5, gap=p.gap)
    else:
        i_e, i_v = list(cv_externa(p.gap).split(p.X_tr))[pliegue]
        idx = np.concatenate([i_e, i_v])  # sin los días de separación
        X = con_rasgos(p.X_tr.iloc[idx], p.tiempo_tr["dia"].iloc[idx], p.y_tr.iloc[idx])
        X_ent, y_ent = X.iloc[: len(i_e)], p.y_tr.iloc[i_e]
        X_val, y_val = X.iloc[len(i_e):], p.y_tr.iloc[i_v]
        cv = cv_interna(p.gap)
    clave = f"retador_{variante}"
    ev = Evaluador(pipeline_retador(variante, seed), X_ent, y_ent, cv, cfg.METRICA_CLF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        OPTIMIZADORES["bayes"](ev, "clf", clave, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = pipeline_retador(variante, seed).set_params(**params)
        t = time.perf_counter()
        final.fit(X_ent, y_ent)
        t_fit = time.perf_counter() - t
        s = final.predict_proba(X_val)[:, 1]
    res = {"variante": variante, "pliegue": pliegue, "seed": seed, "params": params, "puntaje_interno": interno,
           "t_opt": t_opt, "t_fit": t_fit, "historial": ev.historial, "n_ent": len(X_ent), "n_val": len(X_val),
           **metricas_clf(y_val, s, (s >= 0.5).astype(int), True),
           "oof": {"indice": X_val.index.to_numpy(), "y": y_val.to_numpy(), "puntaje": s}}
    if pliegue == -1:
        res["pipeline"] = final
    if guardar:
        RUTA_RETADOR.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res
