"""Protocolo del modelo original: pipeline, puntaje secuencial, validación anidada, modelo final y controles.

El modelo L2E pasa por el mismo protocolo que los 140 modelos del tercer entregable: misma partición cronológica, mismo
preprocesamiento, mismos pliegues externos (``TimeSeriesSplit`` de 5) e internos (de 3), misma separación entre ajuste
y validación, misma semilla, el mismo optimizador bayesiano (Optuna TPE) y el mismo presupuesto de 20 configuraciones.
La única diferencia es el modo de predicción: en cada bloque de validación el modelo asimila las etiquetas de los días
ya cerrados (un día de demora), que es como operaría en producción.
"""

import pickle
import time
import warnings

import numpy as np
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import config as cfg
from .anidada import metricas_clf, particion
from .datos import cv_externa, cv_interna
from .espacios import ESPACIOS, F
from .modelo_original import LogisticaDosEscalas
from .optimizadores import OPTIMIZADORES, Evaluador
from .preprocesamiento import construir_preprocesador

RUTA_CORRIDAS_ORIGINAL = cfg.RUTA_RESULTADOS / "corridas_original"
RUTA_FINALES_ORIGINAL = cfg.RUTA_RESULTADOS / "finales_original"

_LAM = F(1e-1, 1e3, log=True)
_Q_NIVEL = F(1e-3, 1.0, log=True)
_Q_PEND = F(1e-7, 1e-2, log=True)

# Variante → (parámetros fijos del estimador, espacio de búsqueda). Cada variante de la ablación se optimiza con su
# propio espacio (solo los hiperparámetros que la afectan) y el mismo presupuesto.
VARIANTES = {
    "l2e": ({}, {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL, "modelo__q_pendiente": _Q_PEND}),
    "sin_nivel_latente": ({"nivel_latente": False},
                          {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL, "modelo__q_pendiente": _Q_PEND}),
    "sin_deriva_nivel": ({"nivel_latente": False, "q_nivel": 0.0},
                         {"modelo__lam": _LAM, "modelo__q_pendiente": _Q_PEND}),
    "sin_pendientes": ({"actualizar_pendientes": False}, {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL}),
    "estatico": ({"actualizar_nivel": False, "actualizar_pendientes": False},
                 {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL}),
    "escala_unica": ({"escala_unica": True}, {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL}),
    "sin_incertidumbre": ({"integrar_incertidumbre": False},
                          {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL, "modelo__q_pendiente": _Q_PEND}),
}

# Mejoras evaluadas después del modelo base (sección de mejoras del guion): mismo protocolo y presupuesto
_SEMIVIDA = F(20.0, 2000.0, log=True)
MEJORAS = {
    "m1_splines": ({}, {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL, "modelo__q_pendiente": _Q_PEND}),
    "m2_bayes_empirico": ({"q_nivel_empirico": True}, {"modelo__lam": _LAM, "modelo__q_pendiente": _Q_PEND}),
    "m3_recencia": ({}, {"modelo__lam": _LAM, "modelo__q_nivel": _Q_NIVEL, "modelo__q_pendiente": _Q_PEND,
                         "modelo__semivida": _SEMIVIDA}),
}


def definicion(variante: str):
    """Parámetros fijos y espacio de búsqueda de una variante de la ablación o de una mejora."""
    return {**VARIANTES, **MEJORAS}[variante]

NOMBRE_VARIANTE = {
    "m1_splines": "M1 · efectos no lineales del cliente (splines en age y campaign)",
    "m2_bayes_empirico": "M2 · q_u por Bayes empírico (máxima evidencia de Laplace)",
    "m3_recencia": "M3 · ponderación por recencia en el ajuste (semivida)",
    "l2e": "L2E completo", "sin_nivel_latente": "Sin nivel latente en el ajuste (C1)",
    "sin_deriva_nivel": "Sin deriva del nivel (q_u = 0)", "sin_pendientes": "Sin actualizar pendientes (C3)",
    "estatico": "Sin actualización en línea (C2 y C3)", "escala_unica": "Una sola escala (q_β = q_u)",
    "sin_incertidumbre": "Sin integrar la incertidumbre (C4)",
}


def clave_espacio(variante: str) -> str:
    """Registra el espacio de búsqueda de la variante en ``ESPACIOS`` (lo usan los optimizadores del E3)."""
    clave = f"original_{variante}"
    ESPACIOS[("clf", clave)] = definicion(variante)[1]
    return clave


def con_dia(X, tiempo):
    """Agrega la columna ``dia`` (día de contacto reconstruido) que el modelo L2E usa como índice temporal.

    Los modelos de referencia la ignoran: su ``ColumnTransformer`` solo selecciona las columnas del E3.
    """
    return X.assign(dia=np.asarray(tiempo["dia"]))


def preprocesador_con_tiempo(numericas=None, categoricas=None) -> ColumnTransformer:
    """El preprocesador del tercer entregable más la columna ``dia`` sin transformar, al final."""
    base = construir_preprocesador(numericas, categoricas)
    return ColumnTransformer(base.transformers + [("tiempo", "passthrough", ["dia"])])


def preprocesador_splines() -> ColumnTransformer:
    """Mejora M1: el preprocesador del L2E con *splines* cúbicos para ``age`` y ``log1p(campaign)``.

    La edad tiene una relación en forma de U con la suscripción (estudiantes y jubilados suscriben más) que una
    pendiente lineal no captura. Los nodos van en los cuantiles y la extrapolación es lineal, para no repetir el
    problema de los árboles fuera del rango de entrenamiento.
    """
    from sklearn.preprocessing import FunctionTransformer, SplineTransformer
    from sklearn.pipeline import make_pipeline

    pasos = []
    for nombre, trans, cols in construir_preprocesador().transformers:
        cols = [c for c in cols if c not in ("age", "campaign")]
        if cols:
            pasos.append((nombre, trans, cols))
    spl = dict(degree=3, knots="quantile", extrapolation="linear", include_bias=False)
    pasos += [("spl_age", SplineTransformer(n_knots=5, **spl), ["age"]),
              ("spl_campaign", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                             SplineTransformer(n_knots=4, **spl)), ["campaign"]),
              ("tiempo", "passthrough", ["dia"])]
    return ColumnTransformer(pasos)


PREPROCESADORES = {"m1_splines": preprocesador_splines}


def construir_pipeline_original(variante: str = "l2e", preprocesador=None, **params) -> Pipeline:
    """Pipeline preprocesamiento → L2E con los parámetros fijos de la variante."""
    fijos = definicion(variante)[0]
    if preprocesador is None:
        preprocesador = PREPROCESADORES.get(variante, preprocesador_con_tiempo)()
    pre = preprocesador
    return Pipeline([("pre", pre), ("modelo", LogisticaDosEscalas(**{**fijos, **params}))])


def puntaje_secuencial(pipe, X, y, retardo=None) -> np.ndarray:
    """Probabilidades precuenciales: el modelo asimila las etiquetas de los días cerrados mientras predice."""
    Xt = pipe[:-1].transform(X)
    return pipe[-1].predict_proba_secuencial(Xt, np.asarray(y), retardo=retardo)[:, 1]


def ap_secuencial(estimador, X, y) -> float:
    """Puntuador para ``cross_validate``: AUC-PR de las predicciones precuenciales del bloque de validación."""
    return average_precision_score(y, puntaje_secuencial(estimador, X, y))


def ejecutar_original(variante: str, pliegue: int, seed: int = cfg.SEED, guardar: bool = True) -> dict:
    """Una variante del modelo original en un pliegue externo: optimiza en el bucle interno y evalúa en el externo.

    Devuelve las mismas claves que ``anidada.ejecutar_tarea`` (hiperparámetros, métricas externas, tiempos, historial
    y predicciones fuera de muestra), más ``puntaje_estatico``: las probabilidades del mismo modelo sin asimilar
    etiquetas del bloque de validación.
    """
    sufijo = "" if seed == cfg.SEED else f"_s{seed}"
    ruta = RUTA_CORRIDAS_ORIGINAL / f"{variante}_{pliegue}{sufijo}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    X, y = con_dia(p.X_tr, p.tiempo_tr), p.y_tr
    i_ent, i_val = list(cv_externa(p.gap).split(X))[pliegue]
    X_ent, y_ent, X_val, y_val = X.iloc[i_ent], y.iloc[i_ent], X.iloc[i_val], y.iloc[i_val]
    clave = clave_espacio(variante)
    ev = Evaluador(construir_pipeline_original(variante), X_ent, y_ent, cv_interna(p.gap), ap_secuencial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        info = OPTIMIZADORES["bayes"](ev, "clf", clave, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = construir_pipeline_original(variante).set_params(**params)
        t = time.perf_counter()
        final.fit(X_ent, y_ent)
        t_fit = time.perf_counter() - t
        t = time.perf_counter()
        s = puntaje_secuencial(final, X_val, y_val)
        t_pred = time.perf_counter() - t
        s_est = final.predict_proba(X_val)[:, 1]
    met = metricas_clf(y_val, s, (s >= 0.5).astype(int), True)
    res = {"tarea": "clf", "modelo": "original", "balanceo": "ninguno", "optimizador": "bayes", "variante": variante,
           "pliegue": pliegue, "seed": seed, "params": params, "puntaje_interno": interno, "n_eval": len(ev.historial),
           "t_opt": t_opt, "t_por_eval": float(np.mean([h["t_eval"] for h in ev.historial])), "t_fit": t_fit,
           "t_pred": t_pred, "n_ent": len(i_ent), "n_val": len(i_val), **met, "historial": ev.historial, "info": info,
           "n_iter_newton": int(final[-1].n_iter_), "convergio": bool(final[-1].convergio_),
           "oof": {"indice": X_val.index.to_numpy(), "y": y_val.to_numpy(), "puntaje": s, "pred": (s >= 0.5) * 1.0,
                   "puntaje_estatico": s_est}}
    if guardar:
        RUTA_CORRIDAS_ORIGINAL.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def ajustar_original(variante: str = "l2e", seed: int = cfg.SEED, guardar: bool = True) -> dict:
    """Modelo final: repite la optimización con todo el entrenamiento y evalúa una sola vez en la prueba.

    Igual que los finalistas del tercer entregable (``evaluacion.ajustar_finalista``): validación interna
    ``TimeSeriesSplit`` de 5 pliegues sobre todo el entrenamiento, mismo optimizador y presupuesto, reajuste con la
    mejor configuración y predicción precuencial de la prueba con un día de demora en las etiquetas.
    """
    sufijo = "" if seed == cfg.SEED else f"_s{seed}"
    ruta = RUTA_FINALES_ORIGINAL / f"{variante}{sufijo}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    X_tr, X_te = con_dia(p.X_tr, p.tiempo_tr), con_dia(p.X_te, p.tiempo_te)
    clave = clave_espacio(variante)
    ev = Evaluador(construir_pipeline_original(variante), X_tr, p.y_tr, TimeSeriesSplit(n_splits=5, gap=p.gap),
                   ap_secuencial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        OPTIMIZADORES["bayes"](ev, "clf", clave, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = construir_pipeline_original(variante).set_params(**params)
        t = time.perf_counter()
        final.fit(X_tr, p.y_tr)
        t_fit = time.perf_counter() - t
        Xt_te = final[:-1].transform(X_te)
        t = time.perf_counter()
        proba, traza = final[-1].predict_proba_secuencial(Xt_te, p.y_te.to_numpy(), devolver_traza=True)
        t_pred = time.perf_counter() - t
        s_te = proba[:, 1]
        s_est = final.predict_proba(X_te)[:, 1]
        s_tr = final.predict_proba(X_tr)[:, 1]
    res = {"tarea": "clf", "modelo": "original", "balanceo": "ninguno", "optimizador": "bayes", "variante": variante,
           "seed": seed, "params": params, "puntaje_interno": interno, "t_opt": t_opt, "t_fit": t_fit, "t_pred": t_pred,
           "pipeline": final, "es_proba": True, "puntaje_te": s_te, "pred_te": (s_te >= 0.5) * 1.0,
           "puntaje_estatico_te": s_est, "puntaje_tr": s_tr, "traza": traza, "historial": ev.historial,
           **metricas_clf(p.y_te, s_te, (s_te >= 0.5).astype(int), True)}
    if guardar:
        RUTA_FINALES_ORIGINAL.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def em_saerens(prob, prevalencia_ajuste, max_iter=200, tol=1e-8) -> np.ndarray:
    """Ajuste de probabilidades a una prevalencia nueva sin etiquetas (EM de Saerens et al., 2002; MLLS).

    Itera π ← media de p'(y = 1 | x), con p' ∝ (π / π_aj) p para la clase 1 y ((1 − π) / (1 − π_aj)) (1 − p) para la 0.
    Garg et al. (2020) muestran que este estimador de máxima verosimilitud es consistente bajo cambio de prevalencia
    cuando el clasificador está calibrado en el dominio de ajuste.
    """
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)
    pi0 = float(np.clip(prevalencia_ajuste, 1e-6, 1 - 1e-6))
    pi = pi0
    for _ in range(max_iter):
        a = pi / pi0 * p
        b = (1 - pi) / (1 - pi0) * (1 - p)
        p_aj = a / (a + b)
        nuevo = float(p_aj.mean())
        if abs(nuevo - pi) < tol:
            pi = nuevo
            break
        pi = nuevo
    a = pi / pi0 * p
    return a / (a + (1 - pi) / (1 - pi0) * (1 - p))


def em_por_dia(prob, dias, prevalencia_ajuste) -> np.ndarray:
    """Aplica ``em_saerens`` a cada día por separado (control sin etiquetas)."""
    prob, dias = np.asarray(prob, dtype=float), np.asarray(dias)
    salida = np.empty_like(prob)
    for d in np.unique(dias):
        m = dias == d
        salida[m] = em_saerens(prob[m], prevalencia_ajuste)
    return salida


def logit_seguro(p):
    """Logit con recorte, para usar probabilidades de otros modelos como término fijo."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


__all__ = ["VARIANTES", "NOMBRE_VARIANTE", "con_dia", "preprocesador_con_tiempo", "construir_pipeline_original",
           "puntaje_secuencial", "ap_secuencial", "ejecutar_original", "ajustar_original", "em_saerens", "em_por_dia",
           "logit_seguro", "expit"]
