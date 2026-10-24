"""Evaluación final en el conjunto de prueba, calibración y diagnóstico de residuos (sección 5 de la guía)."""

import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from . import config as cfg
from .anidada import metricas_clf, metricas_reg, particion, puntajes
from .modelos import construir_pipeline
from .optimizadores import OPTIMIZADORES, Evaluador

RUTA_FINALES = cfg.RUTA_RESULTADOS / "finales"


def elegir_finalistas(maestra: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Mejor combinación (balanceo, optimizador) de cada modelo según la métrica media del bucle externo.

    Clasificación: mayor AUC-PR media. Regresión: menor RMSE medio.
    """
    t = maestra[maestra["tarea"] == tarea]
    col, asc = ("auc_pr_media", False) if tarea == "clf" else ("rmse_media", True)
    return t.sort_values(col, ascending=asc).groupby("modelo", sort=False).head(1).reset_index(drop=True)


def ajustar_finalista(tarea, modelo, balanceo, optimizador, seed=cfg.SEED, guardar=True) -> dict:
    """Repite la optimización con todo el entrenamiento y evalúa una sola vez en prueba.

    Los hiperparámetros del bucle externo cambian entre pliegues, así que el modelo final se obtiene corriendo el
    mismo optimizador sobre todo el entrenamiento (validación interna ``TimeSeriesSplit`` de 5 pliegues, igual que
    en el Entregable 2) y reajustando con la mejor configuración.
    """
    ruta = RUTA_FINALES / f"{tarea}_{modelo}_{balanceo}_{optimizador}_s{seed}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    y_tr, y_te = (p.y_tr, p.y_te) if tarea == "clf" else (p.r_tr, p.r_te)
    pipe = construir_pipeline(tarea, modelo, balanceo, seed)
    cv = TimeSeriesSplit(n_splits=5, gap=p.gap)
    ev = Evaluador(pipe, p.X_tr, y_tr, cv, cfg.METRICA_CLF if tarea == "clf" else cfg.METRICA_REG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        OPTIMIZADORES[optimizador](ev, tarea, modelo, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = construir_pipeline(tarea, modelo, balanceo, seed).set_params(**params)
        t = time.perf_counter()
        final.fit(p.X_tr, y_tr)
        t_fit = time.perf_counter() - t
        if tarea == "clf":
            s_te, es_proba = puntajes(final, p.X_te)
            s_tr, _ = puntajes(final, p.X_tr)
            pred_te = final.predict(p.X_te)
            met = metricas_clf(y_te, s_te, pred_te, es_proba)
        else:
            s_te = pred_te = final.predict(p.X_te)
            s_tr = final.predict(p.X_tr)
            es_proba = False
            met = metricas_reg(y_te, pred_te)
    res = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador, "seed": seed,
           "params": params, "puntaje_interno": interno, "t_opt": t_opt, "t_fit": t_fit, "pipeline": final,
           "es_proba": es_proba, "puntaje_te": np.asarray(s_te, dtype=float), "pred_te": np.asarray(pred_te, dtype=float),
           "puntaje_tr": np.asarray(s_tr, dtype=float), **met}
    if guardar:
        RUTA_FINALES.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def modelo_referencia_e2(guardar=True) -> dict:
    """Modelo base del Entregable 2: logística L2 con C = 0,01, sin balanceo, ajustada con todo el entrenamiento."""
    ruta = RUTA_FINALES / "clf_referencia_e2.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    final = construir_pipeline("clf", "logistica", "ninguno").set_params(modelo__C=0.01, modelo__l1_ratio=0.0)
    final.set_params(modelo__solver="lbfgs", modelo__max_iter=5000, modelo__tol=1e-4)
    final.fit(p.X_tr, p.y_tr)
    s_te = final.predict_proba(p.X_te)[:, 1]
    res = {"tarea": "clf", "modelo": "referencia_e2", "balanceo": "ninguno", "optimizador": "—",
           "params": {"modelo__C": 0.01, "modelo__l1_ratio": 0.0}, "pipeline": final, "es_proba": True,
           "puntaje_te": s_te, "pred_te": final.predict(p.X_te).astype(float),
           "puntaje_tr": final.predict_proba(p.X_tr)[:, 1],
           **metricas_clf(p.y_te, s_te, final.predict(p.X_te), True)}
    if guardar:
        RUTA_FINALES.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


# ------------------------------------------------------------------------------------------------- Calibración

def ece(y, p, n_bins: int = 10) -> float:
    """Error de calibración esperado con *bins* de igual ancho: Σ (n_b / n) · |frecuencia_b − confianza_b|."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def curva_confiabilidad(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Frecuencia observada frente a probabilidad media predicha por *bin* (estrategia de igual ancho)."""
    fo, pm = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"prob_media": pm, "frecuencia": fo})


def recalibrar(finalista: dict, frac_calibracion: float = 0.2) -> dict:
    """Recalibra un finalista con Platt (sigmoide) e isotónica, sin tocar los datos de prueba.

    El modelo se reajusta con el primer 80 % del entrenamiento (en orden temporal) y el calibrador se ajusta con el
    20 % final; así el calibrador ve probabilidades fuera de muestra, como en uso real. Devuelve las probabilidades de
    prueba sin recalibrar (del mismo modelo reajustado), con Platt y con regresión isotónica.
    """
    p = particion()
    n_cal = int(np.ceil(frac_calibracion * len(p.X_tr)))
    X_a, y_a = p.X_tr.iloc[:-n_cal - p.gap], p.y_tr.iloc[:-n_cal - p.gap]
    X_c, y_c = p.X_tr.iloc[-n_cal:], p.y_tr.iloc[-n_cal:]
    base = construir_pipeline("clf", finalista["modelo"] if finalista["modelo"] != "referencia_e2" else "logistica",
                              finalista["balanceo"]).set_params(**finalista["params"])
    if finalista["modelo"] == "referencia_e2":
        base.set_params(modelo__solver="lbfgs", modelo__max_iter=5000, modelo__tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base.fit(X_a, y_a)
        salida = {}
        s, es_proba = puntajes(base, p.X_te)
        salida["sin_recalibrar"] = s if es_proba else None
        for metodo, nombre in (("sigmoid", "platt"), ("isotonic", "isotonica")):
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=metodo).fit(X_c, y_c)
            salida[nombre] = cal.predict_proba(p.X_te)[:, 1]
    return salida


def recalibracion_mensual(prob_te, y_te, periodo_te, prob_ultimo_mes, y_ultimo_mes) -> np.ndarray:
    """Desplaza el intercepto cada mes para que la probabilidad media iguale la tasa del mes cerrado anterior.

    Es el método del Entregable 2 (sección 4.7.1) generalizado a cualquier clasificador probabilístico: para el mes m
    se busca delta tal que mean(sigmoide(logit(p) + delta)) en el mes m − 1 iguale la tasa observada en m − 1, y se
    aplica ese delta a las predicciones del mes m. Solo usa etiquetas de meses ya cerrados.
    """
    lp_te = logit(np.clip(np.asarray(prob_te, dtype=float), 1e-6, 1 - 1e-6))
    salida = np.empty_like(lp_te)
    periodos = pd.Series(periodo_te).to_numpy()
    previo_lp = logit(np.clip(np.asarray(prob_ultimo_mes, dtype=float), 1e-6, 1 - 1e-6))
    previo_y = np.asarray(y_ultimo_mes)
    y_te = np.asarray(y_te)
    for per in pd.unique(periodos):
        m = periodos == per
        tasa = np.clip(previo_y.mean(), 1e-4, 1 - 1e-4)
        delta = _resolver_desplazamiento(previo_lp, tasa)
        salida[m] = expit(lp_te[m] + delta)
        previo_lp, previo_y = lp_te[m], y_te[m]
    return salida


def _resolver_desplazamiento(lp, tasa, iteraciones=60):
    bajo, alto = -15.0, 15.0
    for _ in range(iteraciones):
        medio = (bajo + alto) / 2
        if expit(lp + medio).mean() < tasa:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2


def resumen_calibracion(y, variantes: dict) -> pd.DataFrame:
    """Brier y ECE de cada variante de probabilidades."""
    filas = []
    for nombre, prob in variantes.items():
        if prob is None:
            continue
        filas.append({"variante": nombre, "brier": brier_score_loss(y, prob), "ece": ece(y, prob),
                      "prob_media": float(np.mean(prob)), "tasa_real": float(np.mean(y))})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------- Residuos (regresión)

def diagnostico_residuos(y, pred, n_bds: int = 2000, rezagos=(10, 20)) -> dict:
    """Pruebas sobre los residuos en orden temporal.

    - White (versión especial de Wooldridge): regresa e² sobre ŷ y ŷ²; evita los cientos de productos cruzados de
      las columnas *one-hot* y conserva la hipótesis alternativa de heterocedasticidad ligada a la media.
    - BDS (Brock, Dechert, Scheinkman y LeBaron, 1996) sobre los últimos ``n_bds`` residuos, dimensiones 2 y 3:
      independencia frente a dependencia lineal o no lineal (la prueba es O(n²) en memoria).
    - Ljung-Box en los rezagos indicados (autocorrelación serial).
    - Jarque-Bera y D'Agostino-Pearson (normalidad).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
    from statsmodels.tsa.stattools import bds

    e = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    exog = np.column_stack([np.ones(len(e)), np.asarray(pred, dtype=float)])
    lm, p_white, _, _ = het_white(e, exog)
    estad_bds, p_bds = bds(e[-n_bds:], max_dim=3)
    lb = acorr_ljungbox(e, lags=list(rezagos))
    jb = stats.jarque_bera(e)
    k2 = stats.normaltest(e)
    return {"white_lm": float(lm), "p_white": float(p_white), "bds": [float(v) for v in np.atleast_1d(estad_bds)],
            "p_bds": [float(v) for v in np.atleast_1d(p_bds)],
            "ljung_box": lb.reset_index().rename(columns={"index": "rezago"}).to_dict("records"),
            "jarque_bera": float(jb.statistic), "p_jarque_bera": float(jb.pvalue), "p_dagostino": float(k2.pvalue),
            "asimetria": float(stats.skew(e)), "curtosis": float(stats.kurtosis(e))}
