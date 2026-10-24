"""Condición R3 para el retador XGB-N, con la misma batería que el L2E (cuaderno, sección 6.7).

Frente a la mejor combinación benchmark (Naive Bayes) y a los demás benchmarks: Nemenyi sobre los meses de prueba
(9 modelos), Wilcoxon pareado por meses, bootstrap estacionario por días y permutación pareada por días, con Holm
sobre los 8 benchmarks. Se repite para el L2E como control de que el cálculo reproduce el del cuaderno.

Uso (desde la raíz del proyecto, después de scripts/ejecutar_retador.py):
    python scripts/r3_retador.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm import estadistica as st  # noqa: E402
from bm import estadistica_e4 as st4  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.retador import RUTA_RETADOR  # noqa: E402

BENCH = ["naive_bayes", "logistica", "bosque", "svm", "xgboost", "arbol", "knn", "referencia_e2"]


def bateria(candidato, puntajes, y, dias, meses):
    M = pd.DataFrame({k: st4.metrica_por_bloque(y, s, meses) for k, s in puntajes.items()}).dropna()
    nem = sp.posthoc_nemenyi_friedman(M.to_numpy())
    nem.index = nem.columns = M.columns
    largo = st4._largo_politis_white(y, dias)
    bt = st4.bootstrap_dias(y, puntajes, dias, candidato, average_precision_score, 1000, largo).set_index("modelo")
    filas = []
    for k in BENCH:
        filas.append({"Benchmark": k, "Δ AUC-PR prueba": average_precision_score(y, puntajes[candidato])
                      - average_precision_score(y, puntajes[k]),
                      "Nemenyi (meses)": nem.loc[candidato, k],
                      "Wilcoxon (meses)": stats.wilcoxon(M[candidato], M[k]).pvalue,
                      "Bootstrap por días": bt.loc[k, "p"],
                      "Permutación por días": st4.permutacion_dias(y, puntajes[candidato], puntajes[k], dias)["p"],
                      "δ de Cliff (meses)": st.delta_cliff(M[candidato], M[k])})
    t = pd.DataFrame(filas).set_index("Benchmark")
    for c in ("Wilcoxon (meses)", "Bootstrap por días", "Permutación por días"):
        t[c] = st.ajustar_p(t[c])
    pruebas = ["Nemenyi (meses)", "Wilcoxon (meses)", "Bootstrap por días", "Permutación por días"]
    # Solo cuenta un rechazo si el candidato es mejor (Δ > 0)
    t["Pruebas que rechazan a favor"] = ((t[pruebas] < 0.05).sum(axis=1) * (t["Δ AUC-PR prueba"] > 0)).astype(int)
    return t


if __name__ == "__main__":
    p = particion()
    y, dias, meses = p.y_te.to_numpy(), p.tiempo_te["dia"].to_numpy(), p.tiempo_te["periodo"].to_numpy()
    with open(cfg.RUTA_RESULTADOS / "e3" / "finales_clf.pkl", "rb") as f:
        fin = pickle.load(f)
    bench = {k: np.asarray(fin[f"clf_{k}"]["puntaje_te"]) for k in BENCH}
    with open(cfg.RUTA_RESULTADOS / "finales_original" / "l2e.pkl", "rb") as f:
        l2e = pickle.load(f)["puntaje_te"]
    with open(RUTA_RETADOR / "xgb_n_prueba.pkl", "rb") as f:
        xgb_n = pickle.load(f)["oof"]["puntaje"]
    salida = {}
    for nombre, s in (("XGB-N", xgb_n), ("L2E", l2e)):
        t = bateria(nombre, {nombre: s, **bench}, y, dias, meses)
        salida[nombre] = t
        print(f"== {nombre} frente a los benchmarks (p ajustados por Holm salvo Nemenyi)")
        print(t.round(4).to_string())
    pd.concat(salida, names=["Candidato"]).to_parquet(RUTA_RETADOR / "r3.parquet")
