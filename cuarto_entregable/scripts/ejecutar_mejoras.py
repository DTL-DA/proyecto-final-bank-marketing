"""Mejoras del modelo original evaluadas con el mismo protocolo (sección de mejoras del guion de sustentación).

- M1, M2 y M3: variantes del L2E en ``protocolo_original.MEJORAS``, con validación anidada (5 pliegues externos,
  Optuna con 20 configuraciones en el bucle interno) y modelo final con evaluación única en prueba.
- M4: ensamble del L2E con un Random Forest reentrenado cada mes con todo lo disponible hasta el mes anterior
  (ventana creciente). En cada pliegue externo, el bosque usa los hiperparámetros que eligió el tercer entregable en
  ese mismo pliegue; el ensamble promedia los rangos normalizados de ambos modelos con peso fijo de 0,5.

Regla de decisión, declarada antes de ejecutar: una mejora se adopta solo si su AUC-PR medio del bucle externo supera
al del L2E en más que la variabilidad entre semillas del optimizador (0,004).

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_mejoras.py [--nucleos 10]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.datos import cv_externa  # noqa: E402
from bm.modelos import construir_pipeline  # noqa: E402
from bm.protocolo_original import MEJORAS, ajustar_original, ejecutar_original  # noqa: E402

DESTINO = cfg.RUTA_RESULTADOS / "mejoras"


def bosque_mensual(X, y, periodos, i_ajuste, i_eval, params):
    """Random Forest reentrenado al inicio de cada mes de ``i_eval`` con ``i_ajuste`` más los meses ya cerrados."""
    salida = np.empty(len(i_eval))
    meses = list(dict.fromkeys(periodos[i_eval]))
    for m in meses:
        en_mes = periodos[i_eval] == m
        previos = i_eval[np.isin(periodos[i_eval], meses[:meses.index(m)])]
        filas = np.r_[i_ajuste, previos]
        modelo = construir_pipeline("clf", "bosque", "ninguno").set_params(**params, modelo__n_jobs=8)
        modelo.fit(X.iloc[filas], y.iloc[filas])
        salida[en_mes] = modelo.predict_proba(X.iloc[i_eval[en_mes]])[:, 1]
    return salida


def rango(s):
    return rankdata(s) / len(s)


def mejora_m4():
    """Ensamble L2E + Random Forest mensual en los pliegues externos y en la prueba."""
    p = particion()
    pl = pd.read_parquet(cfg.RUTA_RESULTADOS / "e3" / "pliegues.parquet")
    pl = pl[(pl["tarea"] == "clf") & (pl["modelo"] == "bosque") & (pl["balanceo"] == "ninguno") &
            (pl["optimizador"] == "bayes")].set_index("pliegue")
    filas = []
    per_tr = p.tiempo_tr["periodo"].to_numpy()
    for k, (i_e, i_v) in enumerate(cv_externa(p.gap).split(p.X_tr)):
        with open(cfg.RUTA_RESULTADOS / "corridas_original" / f"l2e_{k}.pkl", "rb") as f:
            c = pickle.load(f)
        params = json.loads(pl.loc[k, "params"])
        s_rf = bosque_mensual(p.X_tr, p.y_tr, per_tr, i_e, i_v, params)
        y_v = p.y_tr.iloc[i_v].to_numpy()
        assert np.array_equal(c["oof"]["indice"], p.X_tr.index[i_v].to_numpy())
        s_ens = 0.5 * rango(c["oof"]["puntaje"]) + 0.5 * rango(s_rf)
        for nombre, s in (("L2E", c["oof"]["puntaje"]), ("RF mensual", s_rf), ("M4 ensamble", s_ens)):
            filas.append({"conjunto": "externo", "pliegue": k, "modelo": nombre,
                          "auc_pr": average_precision_score(y_v, s), "auc": roc_auc_score(y_v, s)})
        print(f"M4 pliegue {k} listo", flush=True)
    with open(cfg.RUTA_RESULTADOS / "e3" / "finales_clf.pkl", "rb") as f:
        params_te = pickle.load(f)["clf_bosque"]["params"]
    with open(cfg.RUTA_RESULTADOS / "finales_original" / "l2e.pkl", "rb") as f:
        l2e = pickle.load(f)
    X = pd.concat([p.X_tr, p.X_te])
    y = pd.concat([p.y_tr, p.y_te])
    per = np.r_[per_tr, p.tiempo_te["periodo"].to_numpy()]
    n_tr = len(p.X_tr)
    s_rf = bosque_mensual(X, y, per, np.arange(n_tr), np.arange(n_tr, len(X)), params_te)
    s_ens = 0.5 * rango(l2e["puntaje_te"]) + 0.5 * rango(s_rf)
    puntajes = {"L2E": l2e["puntaje_te"], "RF mensual": s_rf, "M4 ensamble": s_ens}
    for nombre, s in puntajes.items():
        filas.append({"conjunto": "prueba", "pliegue": -1, "modelo": nombre,
                      "auc_pr": average_precision_score(p.y_te, s), "auc": roc_auc_score(p.y_te, s)})
    pd.DataFrame(filas).to_parquet(DESTINO / "m4_ensamble.parquet", index=False)
    pd.DataFrame(puntajes).to_parquet(DESTINO / "m4_puntajes_prueba.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    args = ap.parse_args()
    cfg.fijar_semillas()
    DESTINO.mkdir(parents=True, exist_ok=True)
    trabajos = [delayed(ejecutar_original)(v, k) for v in MEJORAS for k in range(cfg.K_EXTERNO)]
    trabajos += [delayed(ajustar_original)(v) for v in MEJORAS]
    Parallel(n_jobs=args.nucleos, backend="loky", verbose=5)(trabajos)
    print("M1, M2 y M3 listos", flush=True)
    mejora_m4()
    print("M4 listo", flush=True)


if __name__ == "__main__":
    main()
