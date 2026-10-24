"""Competencia del modelo retador contra el L2E, con el mismo protocolo (validación anidada y prueba única).

Regla de decisión, declarada antes de ejecutar: el retador reemplaza al L2E solo si su AUC-PR medio del bucle externo
lo supera en más que la variabilidad entre semillas del optimizador (0,004). La prueba se reporta, no decide.

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_retador.py [--nucleos 10] [--variantes xgb_n xgb_t logit_t]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.retador import RUTA_RETADOR, VARIANTES_RETADOR, ejecutar_retador  # noqa: E402

REFERENCIAS = {"l2e": "L2E (modelo original)", "sin_pendientes": "L2E sin actualizar pendientes (ablación)"}


def tarea(variante, pliegue):
    r = ejecutar_retador(variante, pliegue)
    return {k: r[k] for k in ("variante", "pliegue", "auc_pr", "auc", "brier", "t_opt")}


def resumen(filas):
    d = pd.DataFrame(filas)
    for v, nombre in REFERENCIAS.items():
        for k in range(cfg.K_EXTERNO):
            with open(cfg.RUTA_RESULTADOS / "corridas_original" / f"{v}_{k}.pkl", "rb") as f:
                r = pickle.load(f)
            d.loc[len(d)] = {"variante": v, "pliegue": k, "auc_pr": r["auc_pr"], "auc": r["auc"], "brier": r["brier"],
                             "t_opt": r["t_opt"]}
        with open(cfg.RUTA_RESULTADOS / "finales_original" / f"{v}.pkl", "rb") as f:
            r = pickle.load(f)
        d.loc[len(d)] = {"variante": v, "pliegue": -1, "auc_pr": r["auc_pr"], "auc": r["auc"], "brier": r["brier"],
                         "t_opt": r["t_opt"]}
    ext = d[d.pliegue >= 0].groupby("variante")[["auc_pr", "auc"]].agg(["mean", "std"])
    ext.columns = ["AUC-PR externo", "DE", "AUC externo", "DE AUC"]
    te = d[d.pliegue == -1].set_index("variante")[["auc_pr", "auc", "brier"]]
    te.columns = ["AUC-PR prueba", "AUC prueba", "Brier prueba"]
    t = ext.join(te).sort_values("AUC-PR externo", ascending=False)
    t["Δ externo vs L2E"] = t["AUC-PR externo"] - t.loc["l2e", "AUC-PR externo"]
    return d, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES_RETADOR))
    a = ap.parse_args()
    trabajos = [(v, k) for v in a.variantes for k in (-1, *range(cfg.K_EXTERNO))]
    filas = Parallel(n_jobs=a.nucleos, verbose=10)(delayed(tarea)(v, k) for v, k in trabajos)
    d, t = resumen(filas)
    RUTA_RETADOR.mkdir(parents=True, exist_ok=True)
    d.to_parquet(RUTA_RETADOR / "pliegues.parquet")
    t.to_parquet(RUTA_RETADOR / "resumen.parquet")
    print(d[d.pliegue >= 0].pivot_table(index="variante", columns="pliegue", values="auc_pr").round(4).to_string())
    print(t.round(4).to_string())
