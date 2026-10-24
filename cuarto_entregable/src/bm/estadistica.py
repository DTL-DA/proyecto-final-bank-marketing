"""Comparación estadística de modelos (sección 6 de la guía).

Clasificación: Friedman → Nemenyi / diagrama de diferencia crítica → DeLong con Holm-Bonferroni → tamaño de efecto
(delta de Cliff) e intervalos *bootstrap* BCa. Regresión: Model Confidence Set → Giacomini-White / Clark-West →
Diebold-Mariano con corrección HLN y *bootstrap* estacionario → d de Cohen, más la prueba SPA de Hansen.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------------------------------ Friedman y Nemenyi

def friedman(matriz: pd.DataFrame) -> dict:
    """Prueba de Friedman sobre una matriz bloques × modelos (mayor es mejor).

    Parameters
    ----------
    matriz : pandas.DataFrame
        Filas = bloques (pliegues del bucle externo), columnas = modelos, valores = métrica.

    Returns
    -------
    dict
        Estadístico chi cuadrado, grados de libertad, valor p, estadístico F de Iman-Davenport y rangos promedio
        (1 = mejor).
    """
    datos = matriz.dropna(axis=1).to_numpy(dtype=float)
    n, k = datos.shape
    chi2, p = stats.friedmanchisquare(*datos.T)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    rangos = pd.DataFrame(datos, columns=matriz.dropna(axis=1).columns).rank(axis=1, ascending=False).mean()
    return {"chi2": chi2, "gl": k - 1, "p": p, "F_ID": ff, "p_F_ID": p_ff, "n_bloques": n, "k_modelos": k,
            "rangos": rangos.sort_values()}


def diferencia_critica(k: int, n: int, alfa: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi: q_alfa · sqrt(k(k + 1) / (6n)), con q del rango estudentizado / sqrt(2)."""
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


# ------------------------------------------------------------------------------------------------------ DeLong

def _componentes_delong(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    orden = np.argsort(np.r_[pos, neg], kind="mergesort")
    rango_total = stats.rankdata(np.r_[pos, neg])
    rango_pos = stats.rankdata(pos)
    rango_neg = stats.rankdata(neg)
    v10 = (rango_total[:m] - rango_pos) / n
    v01 = 1.0 - (rango_total[m:] - rango_neg) / m
    auc = v10.mean()
    return auc, v10, v01


def delong(y, s1, s2) -> dict:
    """Prueba de DeLong para dos AUC correlacionadas (mismas observaciones).

    Implementación de Sun y Xu (2014) con rangos medios. Devuelve las dos AUC, su diferencia, el error estándar,
    el estadístico z y el valor p bilateral.
    """
    y = np.asarray(y)
    a1, v10_1, v01_1 = _componentes_delong(y, np.asarray(s1, dtype=float))
    a2, v10_2, v01_2 = _componentes_delong(y, np.asarray(s2, dtype=float))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    cov = s10 / len(v10_1) + s01 / len(v01_1)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": a1, "auc_b": a2, "diferencia": a1 - a2, "ee": float(np.sqrt(max(var, 0))), "z": z,
            "p": float(2 * stats.norm.sf(abs(z)))}


def ajustar_p(p, metodo="holm") -> np.ndarray:
    """Valores p ajustados por comparaciones múltiples (``holm`` o ``fdr_bh``)."""
    return multipletests(np.asarray(p, dtype=float), method=metodo)[1]


# --------------------------------------------------------------------------------- Bootstrap y tamaño de efecto

def ic_bootstrap(estadistico, *datos, n_rep=2000, seed=42, nivel=0.95, grupos=100) -> tuple[float, float]:
    """Intervalo *bootstrap* BCa (*bias-corrected and accelerated*, Efron, 1987) de un estadístico pareado.

    La corrección de sesgo z0 sale de la proporción de réplicas por debajo del estimador original; la aceleración
    se estima con un *jackknife* agrupado (se elimina un grupo de observaciones a la vez), que aproxima la del
    *jackknife* completo a una fracción del costo cuando n es de miles.

    Parameters
    ----------
    estadistico : callable
        Función de los arreglos remuestreados, por ejemplo ``lambda y, s: roc_auc_score(y, s)``.
    *datos : array-like
        Arreglos de igual longitud que se remuestrean por filas en conjunto.
    """
    datos = [np.asarray(d) for d in datos]
    n = len(datos[0])
    rng = np.random.default_rng(seed)
    theta = estadistico(*datos)
    reps = np.empty(n_rep)
    for b in range(n_rep):
        i = rng.integers(0, n, n)
        reps[b] = estadistico(*(d[i] for d in datos))
    reps = reps[np.isfinite(reps)]
    z0 = stats.norm.ppf(np.clip(np.mean(reps < theta) + 0.5 * np.mean(reps == theta), 1e-6, 1 - 1e-6))
    etiquetas = rng.permutation(np.arange(n) % grupos)
    jk = np.array([estadistico(*(d[etiquetas != g] for d in datos)) for g in range(grupos)])
    u = jk.mean() - jk
    a = np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5) if np.sum(u ** 2) > 0 else 0.0
    alfas = []
    for q in ((1 - nivel) / 2, 1 - (1 - nivel) / 2):
        z = stats.norm.ppf(q)
        alfas.append(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    return float(np.quantile(reps, alfas[0])), float(np.quantile(reps, alfas[1]))


def delta_cliff(a, b) -> float:
    """Delta de Cliff: P(a > b) − P(a < b), en [−1, 1]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dif = a[:, None] - b[None, :]
    return float((np.sum(dif > 0) - np.sum(dif < 0)) / dif.size)


def magnitud_cliff(d: float) -> str:
    """Umbrales de Romano et al. (2006): 0,147, 0,33 y 0,474."""
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def d_cohen(dif) -> float:
    """d de Cohen para una serie de diferencias de pérdida (media / desviación estándar)."""
    dif = np.asarray(dif, dtype=float)
    return float(dif.mean() / dif.std(ddof=1))


# ---------------------------------------------------------------------------------- Pronóstico (regresión)

def _varianza_hac(x, rezagos):
    x = np.asarray(x, dtype=float) - np.mean(x)
    T = len(x)
    v = np.dot(x, x) / T
    for k in range(1, rezagos + 1):
        v += 2 * (1 - k / (rezagos + 1)) * np.dot(x[k:], x[:-k]) / T
    return v


def diebold_mariano(e1, e2, h: int = 1, potencia: int = 2) -> dict:
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997) para muestras finitas.

    La pérdida es |e|^potencia; d_t = L(e1_t) − L(e2_t) (positivo: el modelo 2 es mejor). La varianza de largo plazo
    usa h − 1 rezagos, como en la prueba original, y el estadístico corregido se compara con una t de T − 1 g. l.
    """
    d = np.abs(np.asarray(e1)) ** potencia - np.abs(np.asarray(e2)) ** potencia
    T = len(d)
    var = _varianza_hac(d, h - 1) / T
    dm = d.mean() / np.sqrt(var)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return {"dif_perdida": float(d.mean()), "DM": float(dm), "DM_HLN": float(hln),
            "p": float(2 * stats.t.sf(abs(hln), T - 1))}


def giacomini_white(l1, l2, condicional: bool = True) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), horizonte 1.

    Instrumentos h_{t−1} = (1, d_{t−1}) en la versión condicional y h = 1 en la incondicional. Estadístico de Wald
    T · Z̄' Ω⁻¹ Z̄ ~ chi cuadrado con q grados de libertad.
    """
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    if condicional:
        H = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        dd = d[1:]
    else:
        H = np.ones((len(d), 1))
        dd = d
    Z = H * dd[:, None]
    T, q = Z.shape
    zbar = Z.mean(axis=0)
    omega = (Z - zbar).T @ (Z - zbar) / T
    est = float(T * zbar @ np.linalg.solve(omega, zbar))
    signo = "modelo 2 mejor" if d.mean() > 0 else "modelo 1 mejor"
    return {"GW": est, "gl": q, "p": float(stats.chi2.sf(est, q)), "dif_media": float(d.mean()), "sentido": signo}


def clark_west(y, pred_restringido, pred_general, rezagos: int | None = None) -> dict:
    """Prueba de Clark y West (2007) para modelos anidados (H0: igual MSPE poblacional; H1: el general es mejor).

    f_t = (y − ŷ_r)² − [(y − ŷ_g)² − (ŷ_r − ŷ_g)²]; se contrasta la media de f con un t unilateral y varianza HAC.
    """
    y, pr, pg = (np.asarray(v, dtype=float) for v in (y, pred_restringido, pred_general))
    f = (y - pr) ** 2 - ((y - pg) ** 2 - (pr - pg) ** 2)
    T = len(f)
    rezagos = int(np.floor(4 * (T / 100) ** (2 / 9))) if rezagos is None else rezagos
    t = f.mean() / np.sqrt(_varianza_hac(f, rezagos) / T)
    return {"CW": float(t), "p": float(stats.norm.sf(t)), "rezagos": rezagos}


def ic_bootstrap_estacionario(dif, n_rep=2000, seed=42, nivel=0.95) -> dict:
    """IC de la media de una serie dependiente con el *bootstrap* estacionario de Politis y Romano (1994).

    El largo medio de bloque se elige con la regla de Politis y White (2004) a través de ``arch``.
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    dif = np.asarray(dif, dtype=float)
    b = float(max(1.0, optimal_block_length(dif)["stationary"].iloc[0]))
    bs = StationaryBootstrap(b, dif, seed=seed)
    medias = np.array([d[0][0].mean() for d in bs.bootstrap(n_rep)])
    a = (1 - nivel) / 2
    return {"media": float(dif.mean()), "ic_inf": float(np.quantile(medias, a)),
            "ic_sup": float(np.quantile(medias, 1 - a)), "bloque": b}


def model_confidence_set(perdidas: pd.DataFrame, alfa: float = 0.10, n_rep: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011) con *bootstrap* estacionario (estadístico T_R).

    Parameters
    ----------
    perdidas : pandas.DataFrame
        Pérdidas por observación (filas en orden temporal) y modelo (columnas).

    Returns
    -------
    pandas.DataFrame
        Pérdida media, rango promedio de pérdida, valor p del MCS y si el modelo queda en el conjunto de confianza.
    """
    from arch.bootstrap import MCS, optimal_block_length

    b = int(np.ceil(np.median([optimal_block_length(perdidas[c].to_numpy())["stationary"].iloc[0]
                               for c in perdidas.columns[:5]])))
    mcs = MCS(perdidas, size=alfa, reps=n_rep, block_size=max(b, 1), method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    pv = mcs.pvalues["Pvalue"]
    salida = pd.DataFrame({"perdida_media": perdidas.mean(), "rango_promedio": perdidas.rank(axis=1).mean(),
                           "p_MCS": pv.reindex(perdidas.columns)})
    salida["en_conjunto"] = salida.index.isin(mcs.included)
    return salida.sort_values("perdida_media")


def spa(perdida_referencia, perdidas: pd.DataFrame, n_rep: int = 1000, seed: int = 42) -> dict:
    """Prueba SPA de Hansen (2005): ¿algún modelo supera a la referencia, controlando el *data snooping*?"""
    from arch.bootstrap import SPA, optimal_block_length

    b = int(np.ceil(optimal_block_length(np.asarray(perdida_referencia))["stationary"].iloc[0]))
    prueba = SPA(np.asarray(perdida_referencia), perdidas.to_numpy(), reps=n_rep, block_size=max(b, 1),
                 bootstrap="stationary", seed=seed)
    prueba.compute()
    return {k: float(v) for k, v in prueba.pvalues.items()}
