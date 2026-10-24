"""Validación cruzada anidada: ejecución de cada combinación, registro estructurado y consolidación."""

import json
import pickle
import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)

from . import config as cfg
from .datos import cv_externa, cv_interna, preparar_particion
from .espacios import ESPACIOS
from .modelos import construir_pipeline
from .optimizadores import OPTIMIZADORES, Evaluador, distribuciones_scipy


@lru_cache(maxsize=1)
def particion():
    """Partición cronológica (se calcula una vez por proceso)."""
    return preparar_particion()


def lista_combinaciones() -> list[tuple]:
    """Todas las combinaciones (tarea, modelo, balanceo, optimizador) del diseño factorial.

    Returns
    -------
    list of tuple
        112 de clasificación (7 × 4 × 4) y 28 de regresión (7 × 4, sin balanceo): 140 en total.
    """
    combos = [("clf", m, b, o) for m in cfg.MODELOS_CLF for b in cfg.BALANCEOS for o in cfg.OPTIMIZADORES]
    combos += [("reg", m, "ninguno", o) for m in cfg.MODELOS_REG for o in cfg.OPTIMIZADORES]
    return combos


def id_tarea(tarea, modelo, balanceo, optimizador, pliegue, seed=cfg.SEED) -> str:
    """Identificador único de una corrida (tarea, modelo, balanceo, optimizador y pliegue); agrega la semilla solo si no es la global."""
    sufijo = "" if seed == cfg.SEED else f"_s{seed}"
    return f"{tarea}_{modelo}_{balanceo}_{optimizador}_{pliegue}{sufijo}"


def puntajes(pipe, X):
    """Puntaje continuo para métricas de ranking: probabilidad si existe, si no la función de decisión."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1], True
        except AttributeError:
            pass
    return pipe.decision_function(X), False


def metricas_clf(y, s, pred, es_proba) -> dict:
    """Métricas de clasificación del bucle externo (umbral por defecto del modelo para las de etiqueta)."""
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "auc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s)}
    if es_proba:
        m["brier"] = brier_score_loss(y, s)
        m["log_loss"] = log_loss(y, np.clip(s, 1e-12, 1 - 1e-12))
    return m


def metricas_reg(r, pred) -> dict:
    """Métricas de regresión en escala log y en segundos."""
    seg, seg_pred = np.expm1(r), np.expm1(pred)
    return {"rmse": root_mean_squared_error(r, pred), "mae": mean_absolute_error(r, pred), "r2": r2_score(r, pred),
            "rmse_seg": root_mean_squared_error(seg, seg_pred), "mae_seg": mean_absolute_error(seg, seg_pred)}


def ejecutar_tarea(tarea, modelo, balanceo, optimizador, pliegue, seed=cfg.SEED, guardar=True) -> dict:
    """Ejecuta una combinación en un pliegue externo: optimiza en el bucle interno y evalúa en el externo.

    Parameters
    ----------
    tarea, modelo, balanceo, optimizador : str
        Identifican la combinación (ver ``lista_combinaciones``).
    pliegue : int
        Índice del pliegue externo (0 a ``K_EXTERNO`` − 1).
    seed : int
        Semilla del modelo, del sobremuestreo y del optimizador.
    guardar : bool
        Si es verdadero, guarda el resultado en ``resultados/corridas`` y lo reutiliza si ya existe.

    Returns
    -------
    dict
        Hiperparámetros elegidos, puntaje interno, métricas externas, tiempos, historial de evaluaciones,
        información propia del optimizador y predicciones fuera de muestra del pliegue externo.
    """
    ruta = cfg.RUTA_CORRIDAS / f"{id_tarea(tarea, modelo, balanceo, optimizador, pliegue, seed)}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    X, y = p.X_tr, (p.y_tr if tarea == "clf" else p.r_tr)
    i_ent, i_val = list(cv_externa(p.gap).split(X))[pliegue]
    X_ent, y_ent, X_val, y_val = X.iloc[i_ent], y.iloc[i_ent], X.iloc[i_val], y.iloc[i_val]
    pipe = construir_pipeline(tarea, modelo, balanceo, seed)
    scoring = cfg.METRICA_CLF if tarea == "clf" else cfg.METRICA_REG
    ev = Evaluador(pipe, X_ent, y_ent, cv_interna(p.gap), scoring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        info = OPTIMIZADORES[optimizador](ev, tarea, modelo, seed=seed)
        t_opt = time.perf_counter() - t
        params, interno = ev.mejor()
        final = pipe.set_params(**params)
        t = time.perf_counter()
        final.fit(X_ent, y_ent)
        t_fit = time.perf_counter() - t
        t = time.perf_counter()
        if tarea == "clf":
            s, es_proba = puntajes(final, X_val)
            pred = final.predict(X_val)
            met = metricas_clf(y_val, s, pred, es_proba)
        else:
            s = pred = final.predict(X_val)
            met = metricas_reg(y_val, pred)
        t_pred = time.perf_counter() - t
    res = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador, "pliegue": pliegue,
           "seed": seed, "params": params, "puntaje_interno": interno, "n_eval": len(ev.historial),
           "t_opt": t_opt, "t_por_eval": float(np.mean([h["t_eval"] for h in ev.historial])), "t_fit": t_fit,
           "t_pred": t_pred, "n_ent": len(i_ent), "n_val": len(i_val), **met,
           "historial": ev.historial, "info": info,
           "oof": {"indice": X_val.index.to_numpy(), "y": y_val.to_numpy(), "puntaje": np.asarray(s, dtype=float),
                   "pred": np.asarray(pred, dtype=float)}}
    if guardar:
        cfg.RUTA_CORRIDAS.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def ejecutar_halving(modelo, pliegue, seed=cfg.SEED, guardar=True, factor=3) -> dict:
    """Successive Halving (multi-fidelidad) en clasificación sin balanceo, con el número de árboles como fidelidad.

    Se parte de 27 configuraciones aleatorias con un noveno del número máximo de árboles; en cada ronda sobrevive
    un tercio (``factor = 3``) y recibe el triple de árboles. El costo se expresa en evaluaciones equivalentes:
    suma de árboles entrenados por configuración dividida por el máximo de árboles.
    """
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    ruta = cfg.RUTA_CORRIDAS / f"{id_tarea('clf', modelo, 'ninguno', 'halving', pliegue, seed)}.pkl"
    if guardar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    p = particion()
    i_ent, i_val = list(cv_externa(p.gap).split(p.X_tr))[pliegue]
    X_ent, y_ent, X_val, y_val = p.X_tr.iloc[i_ent], p.y_tr.iloc[i_ent], p.X_tr.iloc[i_val], p.y_tr.iloc[i_val]
    espacio = ESPACIOS[("clf", modelo)]
    max_r = espacio["modelo__n_estimators"]["alto"]
    min_r = max_r // factor ** 2
    busqueda = HalvingRandomSearchCV(construir_pipeline("clf", modelo, "ninguno", seed),
                                     distribuciones_scipy(espacio, excluir=("modelo__n_estimators",)),
                                     n_candidates=factor ** 3, resource="modelo__n_estimators", min_resources=min_r,
                                     max_resources=max_r, factor=factor, cv=cv_interna(p.gap),
                                     scoring=cfg.METRICA_CLF, random_state=seed, n_jobs=1, refit=False,
                                     error_score=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        busqueda.fit(X_ent, y_ent)
        t_opt = time.perf_counter() - t
        cv = pd.DataFrame(busqueda.cv_results_)
        ultima = cv[cv["iter"] == cv["iter"].max()]
        mejor = ultima.loc[ultima["mean_test_score"].idxmax()]
        params = dict(mejor["params"], modelo__n_estimators=int(mejor["n_resources"]))
        final = construir_pipeline("clf", modelo, "ninguno", seed).set_params(**params).fit(X_ent, y_ent)
        s, es_proba = puntajes(final, X_val)
        met = metricas_clf(y_val, s, final.predict(X_val), es_proba)
    rondas = cv.groupby("iter").agg(candidatos=("params", "size"), arboles=("n_resources", "first"),
                                    mejor=("mean_test_score", "max")).reset_index()
    res = {"tarea": "clf", "modelo": modelo, "balanceo": "ninguno", "optimizador": "halving", "pliegue": pliegue,
           "seed": seed, "params": params, "puntaje_interno": float(mejor["mean_test_score"]),
           "n_eval": int(len(cv)), "eval_equivalentes": float(cv["n_resources"].sum() / max_r), "t_opt": t_opt,
           "t_por_eval": float((cv["mean_fit_time"] + cv["mean_score_time"]).mean() * cfg.K_INTERNO),
           "rondas": rondas.to_dict("records"), "n_ent": len(i_ent), "n_val": len(i_val), **met}
    if guardar:
        cfg.RUTA_CORRIDAS.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(res, f)
    return res


def cargar_corridas(patron="*.pkl") -> list[dict]:
    """Lee todas las corridas guardadas."""
    salida = []
    for ruta in sorted(cfg.RUTA_CORRIDAS.glob(patron)):
        with open(ruta, "rb") as f:
            salida.append(pickle.load(f))
    return salida


def consolidar(corridas: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    """Construye la tabla maestra y las tablas auxiliares a partir de las corridas.

    Returns
    -------
    dict of pandas.DataFrame
        ``pliegues`` (una fila por combinación y pliegue externo), ``maestra`` (una fila por combinación con media y
        desviación estándar de las métricas del bucle externo, tiempos, evaluaciones, hiperparámetros por pliegue y
        semilla), ``historial`` (cada evaluación del bucle interno) y ``oof`` (predicciones fuera de muestra).
    """
    corridas = cargar_corridas() if corridas is None else corridas
    corridas = [c for c in corridas if c["optimizador"] != "halving" and c["seed"] == cfg.SEED]
    claves = ["tarea", "modelo", "balanceo", "optimizador"]
    met_clf = ["accuracy", "precision", "recall", "f1", "auc", "auc_pr", "brier", "log_loss"]
    met_reg = ["rmse", "mae", "r2", "rmse_seg", "mae_seg"]
    filas, hist, oof = [], [], []
    for c in corridas:
        f = {k: c[k] for k in claves + ["pliegue", "seed", "puntaje_interno", "n_eval", "t_opt", "t_por_eval",
                                        "t_fit", "t_pred", "n_ent", "n_val"]}
        f.update({m: c.get(m, np.nan) for m in (met_clf if c["tarea"] == "clf" else met_reg)})
        f["params"] = json.dumps(c["params"], default=str)
        filas.append(f)
        for h in c["historial"]:
            hist.append({**{k: c[k] for k in claves + ["pliegue"]}, "n": h["n"], "puntaje": h["puntaje"],
                         "mejor": h["mejor"], "t_eval": h["t_eval"], "t_acum": h["t_acum"],
                         "params": json.dumps(h["params"], default=str)})
        o = c["oof"]
        oof.append(pd.DataFrame({**{k: c[k] for k in claves + ["pliegue"]}, "indice": o["indice"], "y": o["y"],
                                 "puntaje": o["puntaje"], "pred": o["pred"]}))
    pl = pd.DataFrame(filas)
    metricas = [m for m in met_clf + met_reg if m in pl.columns]
    agg = pl.groupby(claves).agg(
        **{f"{m}_media": (m, "mean") for m in metricas}, **{f"{m}_de": (m, "std") for m in metricas},
        n_pliegues=("pliegue", "size"), n_eval_total=("n_eval", "sum"), t_opt_total=("t_opt", "sum"),
        t_por_eval=("t_por_eval", "mean"), t_fit_medio=("t_fit", "mean"), t_pred_medio=("t_pred", "mean"),
        puntaje_interno_medio=("puntaje_interno", "mean"), seed=("seed", "first")).reset_index()
    agg["params_por_pliegue"] = pl.sort_values("pliegue").groupby(claves)["params"].apply(list).values
    return {"pliegues": pl, "maestra": agg, "historial": pd.DataFrame(hist),
            "oof": pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()}


def guardar_consolidado(tablas: dict[str, pd.DataFrame]) -> None:
    """Guarda la tabla maestra en CSV y parquet, y las auxiliares en parquet."""
    cfg.RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    maestra = tablas["maestra"].copy()
    maestra["params_por_pliegue"] = maestra["params_por_pliegue"].apply(json.dumps)
    maestra.to_csv(cfg.RUTA_RESULTADOS / "tabla_maestra.csv", index=False)
    maestra.to_parquet(cfg.RUTA_RESULTADOS / "tabla_maestra.parquet", index=False)
    for nombre in ("pliegues", "historial", "oof"):
        tablas[nombre].to_parquet(cfg.RUTA_RESULTADOS / f"{nombre}.parquet", index=False)
