"""Ajustes evaluados para los modelos de árboles frente al cambio de régimen (sección de mejoras del guion).

Los árboles no extrapolan fuera del rango de entrenamiento y el 60 % de la prueba tiene un `nr.employed` menor que el
mínimo visto. Se evalúan cinco ajustes con Random Forest y XGBoost de hiperparámetros fijos:

(a) sin variables macroeconómicas ni `month`; (b) macro relativas (cambio frente al mes anterior); (c) híbrido, XGBoost
sobre el logit de la logística C = 0,01; (d) sin macro más el filtro de nivel C2 del L2E; (e) Random Forest reentrenado
cada mes. Cada ajuste se compara con su versión original en dos escenarios: los cinco pliegues externos (solo
entrenamiento) y la prueba, con intervalo y valor p por *bootstrap* estacionario de días.

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_ajustes_arboles.py
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm import estadistica_e4 as st4  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.datos import cv_externa  # noqa: E402
from bm.modelo_original import filtrar_nivel  # noqa: E402
from bm.modelos import construir_pipeline  # noqa: E402
from bm.preprocesamiento import construir_preprocesador  # noqa: E402

MACRO = ["cons.price.idx", "cons.conf.idx", "nr.employed"]
NUM_SIN = [c for c in cfg.VARS_NUM if c not in MACRO]
CAT_SIN = [c for c in cfg.VARS_CAT if c != "month"]
ESTIMADORES = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=8, random_state=cfg.SEED),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, n_jobs=8,
                                     random_state=cfg.SEED, verbosity=0),
}


def pipe(fabrica, numericas=None, categoricas=None):
    return Pipeline([("pre", construir_preprocesador(numericas, categoricas)), ("m", fabrica())])


def macro_relativas(X, periodos):
    """Sustituye cada variable macro por su cambio frente al mes anterior del calendario reconstruido."""
    mensual = X[MACRO].groupby(periodos).first().sort_index().diff().fillna(0.0)
    rel = mensual.reindex(periodos).set_axis(X.index).add_prefix("d_")
    return X.drop(columns=MACRO).join(rel)


def puntajes(X_a, y_a, X_e, per_a, per_e, dias_e, y_e, q_nivel):
    """Puntajes de cada variante en un bloque de evaluación (las que usan etiquetas, solo con días cerrados)."""
    s = {}
    ref = construir_pipeline("clf", "logistica", "ninguno").set_params(
        modelo__C=0.01, modelo__l1_ratio=0.0, modelo__solver="lbfgs", modelo__max_iter=5000).fit(X_a, y_a)
    s["Logística C = 0,01"] = ref.predict_proba(X_e)[:, 1]
    num_rel = NUM_SIN + ["d_" + c for c in MACRO]
    X_rel = macro_relativas(pd.concat([X_a, X_e]), np.r_[per_a, per_e])
    for nombre, fab in ESTIMADORES.items():
        s[f"{nombre} · original"] = pipe(fab).fit(X_a, y_a).predict_proba(X_e)[:, 1]
        s_sin = pipe(fab, NUM_SIN, CAT_SIN).fit(X_a, y_a).predict_proba(X_e)[:, 1]
        s[f"{nombre} · (a) sin macro ni month"] = s_sin
        s[f"{nombre} · (b) macro relativas"] = pipe(fab, num_rel, cfg.VARS_CAT).fit(
            X_rel.loc[X_a.index], y_a).predict_proba(X_rel.loc[X_e.index])[:, 1]
        s[f"{nombre} · (d) sin macro + filtro de nivel C2"] = filtrar_nivel(s_sin, dias_e, y_e, q_nivel)
    pre = construir_preprocesador(NUM_SIN, CAT_SIN).fit(X_a)
    z_a = logit(np.clip(ref.predict_proba(X_a)[:, 1], 1e-6, 1 - 1e-6))
    z_e = logit(np.clip(s["Logística C = 0,01"], 1e-6, 1 - 1e-6))
    hib = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, n_jobs=8, random_state=cfg.SEED, verbosity=0)
    hib.fit(pre.transform(X_a), y_a, base_margin=z_a)
    s["XGBoost · (c) híbrido sobre el logit de la logística"] = hib.predict_proba(pre.transform(X_e), base_margin=z_e)[:, 1]
    # (e) Random Forest reentrenado al inicio de cada mes con todo lo anterior (etiquetas de meses cerrados)
    X_t, y_t = pd.concat([X_a, X_e]), pd.concat([y_a, pd.Series(y_e, index=X_e.index)])
    per_t, n_a = np.r_[per_a, per_e], len(X_a)
    s_e = np.empty(len(X_e))
    for m in dict.fromkeys(per_e):
        en_mes = np.flatnonzero(per_t == m)
        en_mes = en_mes[en_mes >= n_a]
        modelo = pipe(ESTIMADORES["Random Forest"]).fit(X_t.iloc[:en_mes.min()], y_t.iloc[:en_mes.min()])
        s_e[en_mes - n_a] = modelo.predict_proba(X_t.iloc[en_mes])[:, 1]
    s["Random Forest · (e) reentrenado cada mes"] = s_e
    return s


def main():
    p = particion()
    destino = cfg.RUTA_RESULTADOS / "mejoras"
    destino.mkdir(parents=True, exist_ok=True)
    with open(cfg.RUTA_RESULTADOS / "finales_original" / "l2e.pkl", "rb") as f:
        l2e = pickle.load(f)
    q_final = l2e["params"]["modelo__q_nivel"]
    per_tr, per_te = p.tiempo_tr["periodo"].to_numpy(), p.tiempo_te["periodo"].to_numpy()
    dia_tr, dia_te = p.tiempo_tr["dia"].to_numpy(), p.tiempo_te["dia"].to_numpy()
    filas = []
    for k, (i_e, i_v) in enumerate(cv_externa(p.gap).split(p.X_tr)):
        with open(cfg.RUTA_RESULTADOS / "corridas_original" / f"l2e_{k}.pkl", "rb") as f:
            q_k = pickle.load(f)["params"]["modelo__q_nivel"]
        s = puntajes(p.X_tr.iloc[i_e], p.y_tr.iloc[i_e], p.X_tr.iloc[i_v], per_tr[i_e], per_tr[i_v], dia_tr[i_v],
                     p.y_tr.iloc[i_v].to_numpy(), q_k)
        for nombre, v in s.items():
            filas.append({"escenario": "externo", "pliegue": k, "variante": nombre,
                          "auc_pr": average_precision_score(p.y_tr.iloc[i_v], v), "auc": roc_auc_score(p.y_tr.iloc[i_v], v)})
        print(f"pliegue {k} listo", flush=True)
    s = puntajes(p.X_tr, p.y_tr, p.X_te, per_tr, per_te, dia_te, p.y_te.to_numpy(), q_final)
    s["L2E (modelo original)"] = l2e["puntaje_te"]
    yte = p.y_te.to_numpy()
    for nombre, v in s.items():
        filas.append({"escenario": "prueba", "pliegue": -1, "variante": nombre,
                      "auc_pr": average_precision_score(yte, v), "auc": roc_auc_score(yte, v)})
    res = pd.DataFrame(filas)
    # Diferencia de cada ajuste frente a su modelo original en la prueba (bootstrap estacionario por días)
    comp = []
    for nombre in s:
        base = next((f"{m} · original" for m in ESTIMADORES if nombre.startswith(m) and "original" not in nombre), None)
        if base is None:
            continue
        for metrica, f in (("auc_pr", average_precision_score), ("auc", roc_auc_score)):
            b = st4.bootstrap_dias(yte, {"ajuste": s[nombre], "base": s[base]}, dia_te, "ajuste", f, n_rep=1000).iloc[0]
            comp.append({"variante": nombre, "frente_a": base, "metrica": metrica, "diferencia": b["diferencia"],
                         "ic_inf": b["ic_inf"], "ic_sup": b["ic_sup"], "p": b["p"]})
    res.to_parquet(destino / "arboles.parquet", index=False)
    pd.DataFrame(comp).to_parquet(destino / "arboles_bootstrap.parquet", index=False)
    print(res.pivot_table(index="variante", columns="escenario", values="auc_pr").round(4).to_string())


if __name__ == "__main__":
    main()
