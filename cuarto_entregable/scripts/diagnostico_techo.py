"""Diagnóstico de la fiabilidad del L2E: ¿falla el entrenamiento, el algoritmo, el nivel diario o las variables?

Es un análisis de diagnóstico posterior a la evaluación única de la prueba: no elige ni cambia ningún modelo. Las
cotas que usan información no disponible al momento de predecir (etiquetas del mismo día, ``duration`` o entrenar
dentro del periodo de prueba) se marcan como «oráculo» y solo sirven para medir cuánto vale esa información.

1. Sesgo o varianza: AUC y AUC-PR dentro y fuera de muestra con validación aleatoria estratificada (5 pliegues), en
   el periodo de entrenamiento y en el de prueba, para la logística C = 0,01 y un XGBoost de capacidad media.
2. Descomposición del L2E en la prueba: solo el nivel filtrado, solo el efecto del cliente y el modelo completo.
3. Cotas: nivel oráculo del mismo día (tasa de los demás clientes de ese día) más el efecto del cliente; entrenamiento
   ideal en el mismo régimen (validación aleatoria dentro de la prueba); y la información posterior a la llamada
   (``duration``).

Uso (desde la raíz del proyecto):
    python scripts/diagnostico_techo.py
"""

import os
import pickle
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.modelo_original import filtrar_nivel  # noqa: E402
from bm.preprocesamiento import construir_preprocesador  # noqa: E402
from bm.protocolo_original import con_dia  # noqa: E402

DESTINO = cfg.RUTA_RESULTADOS / "diagnostico"


def logistica(numericas=None):
    return Pipeline([("pre", construir_preprocesador(numericas)),
                     ("modelo", LogisticRegression(C=0.01, max_iter=5000))])


def xgboost(numericas=None):
    return Pipeline([("pre", construir_preprocesador(numericas)),
                     ("modelo", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                              colsample_bytree=0.8, tree_method="hist", n_jobs=4,
                                              random_state=cfg.SEED, verbosity=0))])


def metricas(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n = int(0.2 * len(y))
    return {"AUC-PR": average_precision_score(y, s), "AUC": roc_auc_score(y, s), "Prevalencia": y.mean(),
            "Precisión 20 %": y[np.argsort(-s, kind="stable")[:n]].mean()}


def validacion_aleatoria(pipe, X, y):
    """Métricas dentro de muestra (media de los pliegues de ajuste) y fuera de muestra (predicciones fuera de pliegue)."""
    oof = np.empty(len(y))
    dentro = []
    for i_a, i_v in StratifiedKFold(5, shuffle=True, random_state=cfg.SEED).split(X, y):
        m = clone(pipe).fit(X.iloc[i_a], y.iloc[i_a])
        dentro.append(metricas(y.iloc[i_a], m.predict_proba(X.iloc[i_a])[:, 1]))
        oof[i_v] = m.predict_proba(X.iloc[i_v])[:, 1]
    return pd.DataFrame(dentro).mean().to_dict(), metricas(y, oof)


if __name__ == "__main__":
    p = particion()
    y_te = p.y_te.to_numpy()
    dias_te = p.tiempo_te["dia"].to_numpy()

    # 1. Sesgo o varianza
    filas = []
    for periodo, X, y in (("Entrenamiento (2008-2009)", p.X_tr, p.y_tr), ("Prueba (2009-2010)", p.X_te, p.y_te)):
        for nombre, pipe in (("Logística C = 0,01", logistica()), ("XGBoost", xgboost())):
            dentro, fuera = validacion_aleatoria(pipe, X, y)
            filas.append({"Periodo": periodo, "Modelo": nombre, "AUC dentro": dentro["AUC"], "AUC fuera": fuera["AUC"],
                          "AUC-PR dentro": dentro["AUC-PR"], "AUC-PR fuera": fuera["AUC-PR"],
                          "Precisión 20 % fuera": fuera["Precisión 20 %"], "Prevalencia": fuera["Prevalencia"]})
    sesgo = pd.DataFrame(filas)

    # 2. Descomposición del L2E en la prueba
    with open(cfg.RUTA_RESULTADOS / "finales_original" / "l2e.pkl", "rb") as f:
        l2e = pickle.load(f)
    pipe = l2e["pipeline"]
    Xt = pipe[:-1].transform(con_dia(p.X_te, p.tiempo_te))
    proba, traza = pipe[-1].predict_proba_secuencial(Xt, y_te, devolver_traza=True)
    t, F = pipe[-1]._separar(Xt)
    pos = np.searchsorted(traza["dia"], t)
    nivel = traza["nivel"][pos]
    cliente = np.einsum("ij,ij->i", F, traza["pendientes"][pos])

    # 3. Cotas
    d = pd.DataFrame({"dia": dias_te, "y": y_te})
    s_dia, n_dia = d.groupby("dia")["y"].transform("sum").to_numpy(), d.groupby("dia")["y"].transform("size").to_numpy()
    s_otros, n_otros = s_dia - y_te, n_dia - 1  # los demás clientes del mismo día (sin el propio registro)
    nivel_oraculo = np.log((s_otros + 0.5) / (n_otros - s_otros + 0.5))
    _, ideal_log = validacion_aleatoria(logistica(), p.X_te, p.y_te)
    _, ideal_xgb = validacion_aleatoria(xgboost(), p.X_te, p.y_te)
    con_dur = cfg.VARS_NUM + ["log_duration"]
    X_tr_d, X_te_d = p.X_tr.assign(log_duration=p.r_tr), p.X_te.assign(log_duration=p.r_te)
    s_sin = logistica().fit(p.X_tr, p.y_tr).predict_proba(p.X_te)[:, 1]
    s_dur = logistica(con_dur).fit(X_tr_d, p.y_tr).predict_proba(X_te_d)[:, 1]
    q = pipe[-1].q_nivel  # el modelo final usa el q_u elegido por Optuna (sin Bayes empírico)
    filas = [("Azar (prevalencia)", "Disponible", np.random.default_rng(cfg.SEED).random(len(y_te))),
             ("Solo el nivel filtrado del L2E (sin variables del cliente)", "Disponible", nivel),
             ("Solo el efecto del cliente del L2E (sin nivel)", "Disponible", cliente),
             ("L2E completo", "Disponible", proba[:, 1]),
             ("Nivel oráculo del mismo día + efecto del cliente del L2E", "Oráculo", nivel_oraculo + cliente),
             ("Referencia E2 sin duration", "Disponible", s_sin),
             ("Referencia E2 sin duration + filtro de nivel", "Disponible", filtrar_nivel(s_sin, dias_te, y_te, q)),
             ("Referencia E2 con duration", "Oráculo", s_dur),
             ("Referencia E2 con duration + filtro de nivel", "Oráculo", filtrar_nivel(s_dur, dias_te, y_te, q))]
    cotas = pd.DataFrame([{"Escenario": e, "Información": i, **metricas(y_te, s)} for e, i, s in filas])
    for nombre, m in (("Entrenamiento ideal en el mismo régimen: logística (validación aleatoria en la prueba)",
                       ideal_log),
                      ("Entrenamiento ideal en el mismo régimen: XGBoost (validación aleatoria en la prueba)",
                       ideal_xgb)):
        cotas.loc[len(cotas)] = {"Escenario": nombre, "Información": "Oráculo", **m}

    DESTINO.mkdir(parents=True, exist_ok=True)
    sesgo.to_parquet(DESTINO / "sesgo_varianza.parquet")
    cotas.to_parquet(DESTINO / "cotas.parquet")
    pd.set_option("display.width", 250)
    print(sesgo.round(3).to_string(index=False))
    print(cotas.round(3).to_string(index=False))
