"""Tablas resumen de las mejoras y ajustes evaluados después del modelo original (cuaderno, sección 6.13, y guion).

Lee los resultados que escriben ``scripts/ejecutar_mejoras.py`` y ``scripts/ejecutar_ajustes_arboles.py`` y aplica la
regla de decisión declarada antes de ejecutarlos: una mejora se adopta solo si su AUC-PR medio del bucle externo supera
al del L2E en más que la variabilidad entre semillas del optimizador.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as cfg
from .anidada import particion

UMBRAL_SEMILLAS = 0.004  # rango del AUC-PR externo del L2E con las semillas 42, 1, 2 y 3 del optimizador
NOMBRE_MEJORA = {
    "m1_splines": "M1 · Efectos no lineales del cliente (splines en age y campaign)",
    "m2_bayes_empirico": "M2 · q_u por Bayes empírico (máxima evidencia)",
    "m3_recencia": "M3 · Ponderación por recencia en el ajuste",
    "m4_ensamble": "M4 · Ensamble con Random Forest reentrenado cada mes",
    "m5_semillas": "M5 · Promedio de las cuatro semillas del optimizador",
}


def _leer(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


def _corridas(variante, seed=cfg.SEED):
    suf = "" if seed == cfg.SEED else f"_s{seed}"
    return [_leer(cfg.RUTA_RESULTADOS / "corridas_original" / f"{variante}_{k}{suf}.pkl") for k in range(cfg.K_EXTERNO)]


def tabla_mejoras() -> pd.DataFrame:
    """Una fila por mejora (más el L2E de referencia), con el bucle externo, la prueba y la decisión."""
    p = particion()
    res = cfg.RUTA_RESULTADOS
    base = _corridas("l2e")
    ext_base = np.array([c["auc_pr"] for c in base])
    fin_base = _leer(res / "finales_original" / "l2e.pkl")
    filas = [{"clave": "l2e", "Mejora": "L2E (modelo original)", "ext": ext_base,
              "AUC-PR prueba": fin_base["auc_pr"], "AUC prueba": fin_base["auc"]}]
    for v in ("m1_splines", "m2_bayes_empirico", "m3_recencia"):
        ext = np.array([c["auc_pr"] for c in _corridas(v)])
        fin = _leer(res / "finales_original" / f"{v}.pkl")
        filas.append({"clave": v, "Mejora": NOMBRE_MEJORA[v], "ext": ext, "AUC-PR prueba": fin["auc_pr"],
                      "AUC prueba": fin["auc"]})
    m4 = pd.read_parquet(res / "mejoras" / "m4_ensamble.parquet")
    e4 = m4[(m4["conjunto"] == "externo") & (m4["modelo"] == "M4 ensamble")].sort_values("pliegue")["auc_pr"].to_numpy()
    t4 = m4[(m4["conjunto"] == "prueba") & (m4["modelo"] == "M4 ensamble")].iloc[0]
    filas.append({"clave": "m4_ensamble", "Mejora": NOMBRE_MEJORA["m4_ensamble"], "ext": e4,
                  "AUC-PR prueba": t4["auc_pr"], "AUC prueba": t4["auc"]})
    # M5: promedio de las probabilidades de los cuatro L2E (semillas 42, 1, 2 y 3 del optimizador)
    semillas = (cfg.SEED, 1, 2, 3)
    corr = {s: _corridas("l2e", s) for s in semillas}
    e5 = []
    for k in range(cfg.K_EXTERNO):
        y = corr[cfg.SEED][k]["oof"]["y"]
        e5.append(average_precision_score(y, np.mean([corr[s][k]["oof"]["puntaje"] for s in semillas], axis=0)))
    fins = [fin_base] + [_leer(res / "finales_original" / f"l2e_s{s}.pkl") for s in semillas[1:]]
    prom = np.mean([f["puntaje_te"] for f in fins], axis=0)
    filas.append({"clave": "m5_semillas", "Mejora": NOMBRE_MEJORA["m5_semillas"], "ext": np.array(e5),
                  "AUC-PR prueba": average_precision_score(p.y_te, prom), "AUC prueba": roc_auc_score(p.y_te, prom)})
    t = pd.DataFrame(filas)
    t["AUC-PR externo"] = t["ext"].map(np.mean)
    t["DE externo"] = t["ext"].map(lambda a: np.std(a, ddof=1))
    t["Último pliegue"] = t["ext"].map(lambda a: a[-1])
    t["Δ externo frente al L2E"] = t["AUC-PR externo"] - ext_base.mean()
    t["Decisión"] = np.where(t["clave"] == "l2e", "Referencia",
                             np.where(t["Δ externo frente al L2E"] > UMBRAL_SEMILLAS, "Se adopta", "No se adopta"))
    return t.drop(columns="ext")


def tabla_arboles() -> pd.DataFrame:
    """Ajustes para los árboles: bucle externo, prueba y diferencia con su versión original (bootstrap por días)."""
    res = cfg.RUTA_RESULTADOS / "mejoras"
    a = pd.read_parquet(res / "arboles.parquet")
    b = pd.read_parquet(res / "arboles_bootstrap.parquet")
    ext = a[a["escenario"] == "externo"].groupby("variante")["auc_pr"].mean().rename("AUC-PR externo")
    te = a[a["escenario"] == "prueba"].set_index("variante")[["auc_pr", "auc"]].rename(
        columns={"auc_pr": "AUC-PR prueba", "auc": "AUC prueba"})
    bb = b[b["metrica"] == "auc_pr"].set_index("variante")[["diferencia", "ic_inf", "ic_sup", "p"]]
    t = te.join(ext).join(bb)
    orden = [v for v in a[a["escenario"] == "prueba"]["variante"]]
    return t.loc[orden].reset_index()


__all__ = ["tabla_mejoras", "tabla_arboles", "NOMBRE_MEJORA", "UMBRAL_SEMILLAS"]
