"""Análisis complementarios del modelo original: sensibilidad, robustez, complejidad empírica y SHAP.

Todos parten del modelo final L2E (``resultados/finales_original/l2e.pkl``) y escriben en ``resultados/analisis``:

- ``sensibilidad_q.parquet``: rejilla de (q_nivel, q_pendiente) con λ fijo; AUC-PR del bucle externo (sin reoptimizar)
  y de la prueba.
- ``sensibilidad_lam.parquet``: barrido de λ con los q del modelo final.
- ``retardo.parquet``: AUC-PR y AUC en prueba con 1 a 40 días de demora en las etiquetas, para el L2E y para la
  referencia del Entregable 2 con el filtro de nivel.
- ``ruido.parquet`` y ``tamano.parquet``: etiquetas de entrenamiento con ruido y ventanas de entrenamiento más cortas.
- ``complejidad.parquet``: tiempos de ajuste al variar n, p y T, frente a la solución densa del sistema de Newton.
- ``shap_*.parquet``: valores SHAP en escala logit del L2E (exactos), de la referencia del E2 (exactos) y de Naive
  Bayes (permutaciones).

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_analisis.py [--nucleos 10]
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.datos import cv_externa  # noqa: E402
from bm.modelo_original import LogisticaDosEscalas, filtrar_nivel  # noqa: E402
from bm.protocolo_original import con_dia, construir_pipeline_original, logit_seguro, puntaje_secuencial  # noqa: E402

DESTINO = cfg.RUTA_RESULTADOS / "analisis"
Q_NIVEL = np.logspace(-3, 0, 7)
Q_PEND = np.r_[0.0, np.logspace(-7, -2, 6)]
LAMS = np.logspace(-1, 3, 9)
RETARDOS = (1, 2, 3, 5, 10, 20, 40)


def _final():
    with open(cfg.RUTA_RESULTADOS / "finales_original" / "l2e.pkl", "rb") as f:
        return pickle.load(f)


def _params(fin):
    return {k.split("__")[1]: v for k, v in fin["params"].items()}


def evaluar_config(params: dict) -> dict:
    """AUC-PR medio del bucle externo (ajuste en cada pliegue, sin reoptimizar) y AUC-PR de la prueba."""
    p = particion()
    X, Xte = con_dia(p.X_tr, p.tiempo_tr), con_dia(p.X_te, p.tiempo_te)
    ext = []
    for i_e, i_v in cv_externa(p.gap).split(X):
        m = construir_pipeline_original("l2e", **params).fit(X.iloc[i_e], p.y_tr.iloc[i_e])
        ext.append(average_precision_score(p.y_tr.iloc[i_v], puntaje_secuencial(m, X.iloc[i_v], p.y_tr.iloc[i_v])))
    m = construir_pipeline_original("l2e", **params).fit(X, p.y_tr)
    s = puntaje_secuencial(m, Xte, p.y_te)
    return {**params, "auc_pr_externo": float(np.mean(ext)), "auc_pr_prueba": average_precision_score(p.y_te, s),
            "auc_prueba": roc_auc_score(p.y_te, s)}


def retardo(fin, ref) -> pd.DataFrame:
    """Demora de 1 a 40 días en las etiquetas: L2E frente a la referencia del E2 con el filtro de nivel."""
    p = particion()
    Xte = con_dia(p.X_te, p.tiempo_te)
    dias, yte = p.tiempo_te["dia"].to_numpy(), p.y_te.to_numpy()
    filas = []
    for r in RETARDOS:
        s = puntaje_secuencial(fin["pipeline"], Xte, p.y_te, retardo=r)
        s_ref = filtrar_nivel(ref["puntaje_te"], dias, yte, q=fin["params"]["modelo__q_nivel"], retardo=r)
        for nombre, sc in (("L2E", s), ("Referencia E2 + filtro de nivel", s_ref)):
            filas.append({"retardo": r, "modelo": nombre, "auc_pr": average_precision_score(yte, sc),
                          "auc": roc_auc_score(yte, sc)})
    return pd.DataFrame(filas)


def ruido_y_tamano(fin) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustez a etiquetas de entrenamiento ruidosas y a ventanas de entrenamiento más cortas."""
    from sklearn.linear_model import LogisticRegression
    from bm.modelos import construir_pipeline

    p = particion()
    X, Xte = con_dia(p.X_tr, p.tiempo_tr), con_dia(p.X_te, p.tiempo_te)
    params = _params(fin)
    rng = np.random.default_rng(cfg.SEED)

    def referencia(Xa, ya):
        m = construir_pipeline("clf", "logistica", "ninguno").set_params(modelo__C=0.01, modelo__l1_ratio=0.0)
        m.set_params(modelo__solver="lbfgs", modelo__max_iter=5000, modelo__tol=1e-4)
        return m.fit(Xa, ya).predict_proba(p.X_te)[:, 1]

    filas_r = []
    for rho in (0.0, 0.05, 0.10, 0.20):
        voltea = rng.random(len(p.y_tr)) < rho
        y_r = pd.Series(np.where(voltea, 1 - p.y_tr, p.y_tr), index=p.y_tr.index)
        m = construir_pipeline_original("l2e", **params).fit(X, y_r)
        s = puntaje_secuencial(m, Xte, p.y_te)
        voltea_te = rng.random(len(p.y_te)) < rho
        y_te_r = pd.Series(np.where(voltea_te, 1 - p.y_te, p.y_te), index=p.y_te.index)
        s_r = puntaje_secuencial(m, Xte, y_te_r)  # el filtro también asimila etiquetas con ruido
        s_ref = referencia(p.X_tr, y_r)
        filas_r += [{"ruido": rho, "modelo": "L2E (ruido solo en el ajuste)", "auc_pr": average_precision_score(p.y_te, s)},
                    {"ruido": rho, "modelo": "L2E (ruido también en las etiquetas que asimila)",
                     "auc_pr": average_precision_score(p.y_te, s_r)},
                    {"ruido": rho, "modelo": "Referencia E2", "auc_pr": average_precision_score(p.y_te, s_ref)}]
    filas_t = []
    for frac in (0.25, 0.50, 0.75, 1.0):
        n = int(len(X) * frac)
        Xa, ya = X.iloc[-n:], p.y_tr.iloc[-n:]
        m = construir_pipeline_original("l2e", **params).fit(Xa, ya)
        filas_t += [{"fraccion": frac, "registros": n, "modelo": "L2E",
                     "auc_pr": average_precision_score(p.y_te, puntaje_secuencial(m, Xte, p.y_te))},
                    {"fraccion": frac, "registros": n, "modelo": "Referencia E2",
                     "auc_pr": average_precision_score(p.y_te, referencia(p.X_tr.iloc[-n:], ya))}]
    return pd.DataFrame(filas_r), pd.DataFrame(filas_t)


def _newton_denso(F, idx, y, lam, q, iteraciones):
    """Una sola resolución por iteración del sistema de Newton completo (T + p) × (T + p), para comparar costos."""
    T, p = idx.max() + 1, F.shape[1]
    u, b = np.zeros(T), np.zeros(p)
    for _ in range(iteraciones):
        pi = expit(u[idx] + F @ b)
        w = pi * (1 - pi)
        A = np.diag(np.bincount(idx, w, T) + 2 / q) - np.diag(np.full(T - 1, 1 / q), 1) - np.diag(np.full(T - 1, 1 / q), -1)
        B = np.zeros((T, p))
        np.add.at(B, idx, w[:, None] * F)
        H = np.block([[A, B], [B.T, (F * w[:, None]).T @ F + lam * np.eye(p)]])
        g = np.r_[np.bincount(idx, y - pi, T), F.T @ (y - pi) - lam * b]
        paso = np.linalg.solve(H, g)
        u, b = u + paso[:T], b + paso[T:]


def complejidad() -> pd.DataFrame:
    """Tiempo de ajuste al variar n, p y T con datos sintéticos, frente a la solución densa del mismo sistema."""
    rng = np.random.default_rng(cfg.SEED)
    filas = []

    def medir(n, p, T, denso=False, rep=3):
        t_idx = np.sort(rng.integers(0, T, n))
        t_idx[:T] = np.arange(T)
        t_idx.sort()
        F = rng.normal(size=(n, p))
        y = rng.binomial(1, expit(-1 + 0.3 * F[:, 0]))
        X = np.column_stack([F, t_idx])
        tiempos, iters = [], []
        for _ in range(rep):
            t0 = time.perf_counter()
            m = LogisticaDosEscalas(lam=10, q_nivel=0.1).fit(X, y)
            tiempos.append(time.perf_counter() - t0)
            iters.append(m.n_iter_)
        fila = {"n": n, "p": p, "T": T, "t_ajuste": float(np.median(tiempos)), "iteraciones": int(np.median(iters))}
        if denso:
            t0 = time.perf_counter()
            _newton_denso(F, t_idx, y.astype(float), 10.0, 0.1, fila["iteraciones"])
            fila["t_denso"] = time.perf_counter() - t0
        return fila

    for n in (2000, 4000, 8000, 16000, 32000, 64000):
        filas.append({"experimento": "n", **medir(n, 50, 150)})
    for p in (10, 25, 50, 100, 200, 400):
        filas.append({"experimento": "p", **medir(20000, p, 150)})
    for T in (50, 100, 200, 400, 800, 1600, 3200):
        filas.append({"experimento": "T", **medir(20000, 50, T, denso=T <= 1600)})
    return pd.DataFrame(filas)


def shap_l2e(fin) -> pd.DataFrame:
    """SHAP exacto del L2E en escala logit, con las pendientes y el nivel vigentes el día de cada predicción.

    logit_i = base + (nivel_t − base) + Σ_j β_tj (x_ij − x̄_j), donde x̄ es la media de entrenamiento de cada columna
    preprocesada, nivel_t = u_t + β_t·x̄ es el logit esperado del día y base es su promedio en la prueba.
    """
    p = particion()
    pipe = fin["pipeline"]
    pre = pipe[:-1]
    nombres = [c.split("__", 1)[-1] for c in pre.get_feature_names_out()][:-1]
    Zte = pre.transform(con_dia(p.X_te, p.tiempo_te))[:, :-1]
    Ztr = pre.transform(con_dia(p.X_tr, p.tiempo_tr))[:, :-1]
    xbar = Ztr.mean(axis=0)
    tr = fin["traza"]
    pos = {d: i for i, d in enumerate(tr["dia"])}
    k = np.array([pos[d] for d in p.tiempo_te["dia"].to_numpy()])
    B = tr["pendientes"][k]
    nivel = tr["nivel"][k] + B @ xbar
    valores = B * (Zte - xbar)
    base = float(nivel.mean())
    df = pd.DataFrame(valores, columns=nombres, index=p.X_te.index)
    df.insert(0, "nivel latente del día", nivel - base)
    df.attrs["base"] = base
    return df.assign(_base=base)


def shap_referencia(ref) -> pd.DataFrame:
    """SHAP exacto de la logística del Entregable 2 (lineal en el logit): β_j (x_ij − x̄_j)."""
    p = particion()
    pipe = ref["pipeline"]
    pre, mod = pipe.named_steps["pre"], pipe.named_steps["modelo"]
    nombres = [c.split("__", 1)[-1] for c in pre.get_feature_names_out()]
    Zte, Ztr = pre.transform(p.X_te), pre.transform(p.X_tr)
    vals = mod.coef_.ravel() * (Zte - Ztr.mean(axis=0))
    base = float(mod.intercept_[0] + mod.coef_.ravel() @ Ztr.mean(axis=0))
    return pd.DataFrame(vals, columns=nombres, index=p.X_te.index).assign(_base=base)


def shap_naive_bayes(nb, n=300) -> pd.DataFrame:
    """SHAP por permutaciones de Naive Bayes (mejor combinación global del bucle externo), en escala logit."""
    import shap

    p = particion()
    pipe = nb["pipeline"]
    pre, mod = pipe.named_steps["pre"], pipe.named_steps["modelo"]
    nombres = [c.split("__", 1)[-1] for c in pre.get_feature_names_out()]
    Zte = pd.DataFrame(pre.transform(p.X_te), columns=nombres, index=p.X_te.index)
    Ztr = pd.DataFrame(pre.transform(p.X_tr), columns=nombres, index=p.X_tr.index)
    fondo = shap.sample(Ztr, 100, random_state=cfg.SEED)
    muestra = Zte.sample(n, random_state=cfg.SEED)
    f = lambda A: logit_seguro(mod.predict_proba(np.asarray(A))[:, 1])  # noqa: E731
    sv = shap.PermutationExplainer(f, fondo, seed=cfg.SEED)(muestra, silent=True)
    return pd.DataFrame(sv.values, columns=nombres, index=muestra.index).assign(_base=float(np.mean(sv.base_values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    args = ap.parse_args()
    cfg.fijar_semillas()
    DESTINO.mkdir(parents=True, exist_ok=True)
    fin = _final()
    with open(cfg.RUTA_RESULTADOS / "e3" / "finales_clf.pkl", "rb") as f:
        e3 = pickle.load(f)
    base = _params(fin)
    rejilla = [{**base, "q_nivel": float(a), "q_pendiente": float(b)} for a in Q_NIVEL for b in Q_PEND]
    lams = [{**base, "lam": float(v)} for v in LAMS]
    res = Parallel(n_jobs=args.nucleos, backend="loky", verbose=2)(delayed(evaluar_config)(c) for c in rejilla + lams)
    pd.DataFrame(res[:len(rejilla)]).to_parquet(DESTINO / "sensibilidad_q.parquet", index=False)
    pd.DataFrame(res[len(rejilla):]).to_parquet(DESTINO / "sensibilidad_lam.parquet", index=False)
    print("Sensibilidad lista", flush=True)
    retardo(fin, e3["clf_referencia_e2"]).to_parquet(DESTINO / "retardo.parquet", index=False)
    r, t = ruido_y_tamano(fin)
    r.to_parquet(DESTINO / "ruido.parquet", index=False)
    t.to_parquet(DESTINO / "tamano.parquet", index=False)
    print("Retardo, ruido y tamaño listos", flush=True)
    shap_l2e(fin).to_parquet(DESTINO / "shap_l2e.parquet")
    shap_referencia(e3["clf_referencia_e2"]).to_parquet(DESTINO / "shap_referencia.parquet")
    shap_naive_bayes(e3["clf_naive_bayes"]).to_parquet(DESTINO / "shap_naive_bayes.parquet")
    print("SHAP listo", flush=True)
    complejidad().to_parquet(DESTINO / "complejidad.parquet", index=False)
    print("Complejidad lista", flush=True)


if __name__ == "__main__":
    main()
