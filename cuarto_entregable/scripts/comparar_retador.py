"""Pruebas pareadas del retador (y de la ablación sin pendientes) contra el L2E.

Bucle externo (decide): pliegues ganados, t de Nadeau y Bengio y prueba bayesiana de rangos con signo (ROPE 0,01).
Prueba (solo se reporta): diferencia de AUC-PR con bootstrap estacionario por días, permutación pareada por días,
Wilcoxon sobre el AUC-PR mensual y precisión en el 20 % superior.

Uso (desde la raíz del proyecto, después de scripts/ejecutar_retador.py):
    python scripts/comparar_retador.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.estadistica_e4 import (bootstrap_dias, metrica_por_bloque, permutacion_dias, rangos_signo_bayes,  # noqa: E402
                               t_nadeau_bengio)
from bm.retador import RUTA_RETADOR, VARIANTES_RETADOR  # noqa: E402

NOMBRES = {"xgb_n": "XGB-N (retador)", "xgb_t": "XGBoost con rezagos", "logit_t": "Logística con rezagos",
           "sin_pendientes": "L2E sin actualizar pendientes"}


def cargar(v, k):
    if v in VARIANTES_RETADOR:
        with open(RUTA_RETADOR / f"{v}_{'prueba' if k == -1 else k}.pkl", "rb") as f:
            r = pickle.load(f)
        return r, (r["oof"]["puntaje"] if k == -1 else None)
    carpeta = "finales_original" if k == -1 else "corridas_original"
    with open(cfg.RUTA_RESULTADOS / carpeta / f"{v}{'' if k == -1 else f'_{k}'}.pkl", "rb") as f:
        r = pickle.load(f)
    return r, (r["puntaje_te"] if k == -1 else None)


def precision_superior(y, s, k=0.2):
    return float(np.asarray(y)[np.argsort(-np.asarray(s), kind="stable")[: int(len(y) * k)]].mean())


if __name__ == "__main__":
    p = particion()
    y, dias, meses = p.y_te.to_numpy(), p.tiempo_te["dia"].to_numpy(), p.tiempo_te["periodo"].to_numpy()
    ref = [cargar("l2e", k)[0] for k in range(cfg.K_EXTERNO)]
    a_ref = np.array([r["auc_pr"] for r in ref])
    s_ref = cargar("l2e", -1)[1]
    m_ref = metrica_por_bloque(y, s_ref, meses)
    puntajes = {v: cargar(v, -1)[1] for v in NOMBRES}
    boot = bootstrap_dias(y, {**puntajes, "l2e": s_ref}, dias, "l2e").set_index("modelo")  # diferencia = L2E − modelo
    filas = []
    for v, nombre in NOMBRES.items():
        a = np.array([cargar(v, k)[0]["auc_pr"] for k in range(cfg.K_EXTERNO)])
        nb = t_nadeau_bengio(a, a_ref, [r["n_ent"] for r in ref], [r["n_val"] for r in ref])
        bayes = rangos_signo_bayes(a, a_ref)
        m = metrica_por_bloque(y, puntajes[v], meses).reindex(m_ref.index)
        filas.append({"Modelo": nombre, "clave": v, "Δ AUC-PR externo": a.mean() - a_ref.mean(),
                      "Pliegues ganados": int((a > a_ref).sum()), "p Nadeau-Bengio": nb["p"],
                      "P(L2E mejor)": bayes["p_b_mejor"], "P(equivalentes)": bayes["p_rope"],
                      "P(modelo mejor)": bayes["p_a_mejor"],
                      "Δ AUC-PR prueba": -boot.loc[v, "diferencia"],
                      "p bootstrap días": boot.loc[v, "p"],
                      "p permutación días": permutacion_dias(y, puntajes[v], s_ref, dias)["p"],
                      "Meses ganados": f"{int((m > m_ref).sum())} de {len(m_ref)}",
                      "p Wilcoxon meses": wilcoxon(m, m_ref).pvalue,
                      "Precisión 20 %": precision_superior(y, puntajes[v])})
    t = pd.DataFrame(filas)
    t.loc[len(t)] = {"Modelo": "L2E (referencia)", "clave": "l2e", "Precisión 20 %": precision_superior(y, s_ref)}
    t.to_parquet(RUTA_RETADOR / "pruebas.parquet")
    print(boot.round(4).to_string())
    print(t.drop(columns="clave").round(4).T.to_string())
