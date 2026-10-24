"""Pruebas estadísticas nuevas del cuarto entregable (bloque D de la guía).

Complementan ``estadistica.py`` (Friedman, Nemenyi, DeLong, BCa, delta de Cliff, Diebold-Mariano) con:

- métricas por bloque temporal (mes o semana) para dar potencia a las pruebas de rangos;
- la prueba t corregida de Nadeau y Bengio (2003) sobre los pliegues externos;
- la prueba bayesiana de rangos con signo con región de equivalencia práctica (ROPE; Benavoli et al., 2017);
- un *bootstrap* estacionario por días (Politis y Romano, 1994) para la diferencia de AUC-PR o AUC entre dos modelos
  en la prueba, que respeta la dependencia entre registros del mismo día y entre días consecutivos.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


def metrica_por_bloque(y, puntaje, bloques, metrica=average_precision_score, min_registros=100) -> pd.Series:
    """Métrica de un modelo en cada bloque (mes, semana) con al menos ``min_registros`` y ambas clases."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(puntaje, dtype=float), "b": np.asarray(bloques)})
    salida = {}
    for b, g in d.groupby("b", sort=True):
        if len(g) >= min_registros and 0 < g["y"].sum() < len(g):
            salida[b] = metrica(g["y"], g["s"])
    return pd.Series(salida, dtype=float)


def t_nadeau_bengio(a, b, n_ajuste, n_validacion) -> dict:
    """Prueba t pareada con la varianza corregida de Nadeau y Bengio (2003) para pliegues no independientes.

    t = d̄ / sqrt((1/k + n_val/n_aj) · s²_d), con k − 1 grados de libertad. Con ``TimeSeriesSplit`` el ajuste crece
    entre pliegues, así que se usa la razón media n_val / n_aj.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    razon = float(np.mean(np.asarray(n_validacion, dtype=float) / np.asarray(n_ajuste, dtype=float)))
    var = (1.0 / k + razon) * np.var(d, ddof=1)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return {"t": float(t), "gl": k - 1, "p": float(2 * stats.t.sf(abs(t), k - 1)), "dif_media": float(d.mean()),
            "factor_correccion": float(1 + k * razon)}


def rangos_signo_bayes(a, b, rope=0.01, prior=0.5, n_muestras=50_000, seed=42) -> dict:
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017) con proceso de Dirichlet.

    Se agrega una pseudoobservación z₀ = 0 con peso ``prior``; para cada muestra w ~ Dirichlet(prior, 1, ..., 1) se
    calculan θ_izq = Σ w_i w_j 1[z_i + z_j < −2r], θ_rope y θ_der con los promedios de Walsh. Devuelve la probabilidad
    posterior de que cada región sea la más probable: P(a < b), P(equivalencia práctica) y P(a > b).
    """
    z = np.r_[0.0, np.asarray(a, dtype=float) - np.asarray(b, dtype=float)]
    pesos = np.r_[prior, np.ones(len(z) - 1)]
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(pesos, n_muestras)
    suma = z[:, None] + z[None, :]
    izq, der = (suma < -2 * rope).astype(float), (suma > 2 * rope).astype(float)
    t_izq = np.einsum("si,ij,sj->s", w, izq, w)
    t_der = np.einsum("si,ij,sj->s", w, der, w)
    t_rope = 1.0 - t_izq - t_der
    gana = np.argmax(np.column_stack([t_izq, t_rope, t_der]), axis=1)
    return {"p_b_mejor": float(np.mean(gana == 0)), "p_rope": float(np.mean(gana == 1)),
            "p_a_mejor": float(np.mean(gana == 2)), "rope": rope}


def _indices_estacionarios(T, largo_medio, rng):
    """Una réplica del *bootstrap* estacionario: bloques de largo geométrico con media ``largo_medio`` (circular)."""
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    nuevo = rng.random(T) < 1.0 / largo_medio
    saltos = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = saltos[t] if nuevo[t] else (idx[t - 1] + 1) % T
    return idx


def _largo_politis_white(y, dias):
    from arch.bootstrap import optimal_block_length
    tasa = pd.Series(np.asarray(y)).groupby(np.asarray(dias)).mean().to_numpy()
    return float(max(1.0, optimal_block_length(tasa)["stationary"].iloc[0]))


def ic_metrica_dias(y, puntaje, dias, metrica=average_precision_score, n_rep=1000, largo_medio=None, seed=42):
    """Intervalo percentil al 95 % de una métrica con *bootstrap* estacionario por días."""
    y, s, dias = np.asarray(y), np.asarray(puntaje, dtype=float), np.asarray(dias)
    unicos = np.unique(dias)
    filas_dia = [np.flatnonzero(dias == d) for d in unicos]
    largo_medio = _largo_politis_white(y, dias) if largo_medio is None else largo_medio
    rng = np.random.default_rng(seed)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        sel = np.concatenate([filas_dia[i] for i in _indices_estacionarios(len(unicos), largo_medio, rng)])
        reps[r] = metrica(y[sel], s[sel])
    return float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def permutacion_dias(y, puntaje_a, puntaje_b, dias, metrica=average_precision_score, n_rep=1000, seed=42) -> dict:
    """Prueba de permutación pareada por días para la diferencia de una métrica entre dos modelos.

    Bajo H0 los dos modelos son intercambiables; cada réplica intercambia, día por día y con probabilidad 1/2, los
    puntajes de los dos modelos (convertidos a rangos normalizados para que las escalas sean comparables). Al
    permutar días completos se conserva la dependencia dentro del día y la serie de días queda intacta.
    """
    y, dias = np.asarray(y), np.asarray(dias)
    ra = stats.rankdata(puntaje_a) / len(y)
    rb = stats.rankdata(puntaje_b) / len(y)
    unicos, pos = np.unique(dias, return_inverse=True)
    obs = metrica(y, ra) - metrica(y, rb)
    rng = np.random.default_rng(seed)
    extremos = 0
    for _ in range(n_rep):
        cambia = (rng.random(len(unicos)) < 0.5)[pos]
        a, b = np.where(cambia, rb, ra), np.where(cambia, ra, rb)
        extremos += abs(metrica(y, a) - metrica(y, b)) >= abs(obs) - 1e-12
    return {"diferencia": float(obs), "p": float((extremos + 1) / (n_rep + 1)), "n_rep": n_rep}


def bootstrap_dias(y, puntajes: dict, dias, referencia, metrica=average_precision_score, n_rep=1000,
                   largo_medio=None, seed=42) -> pd.DataFrame:
    """Diferencia de una métrica entre ``referencia`` y cada modelo de ``puntajes``, con *bootstrap* estacionario por días.

    Se remuestrean días completos (todos sus registros), en bloques de días consecutivos, de modo que la réplica conserva
    la dependencia dentro del día (el mismo nivel latente) y entre días cercanos. Para cada comparación devuelve la
    diferencia observada, el intervalo percentil al 95 % y el valor p bilateral 2 · mín(P*(Δ ≤ 0), P*(Δ ≥ 0)).

    Parameters
    ----------
    largo_medio : float, optional
        Largo medio de bloque en días. Por defecto, la regla de Politis y White (2004) sobre la serie de la tasa diaria.
    """
    y = np.asarray(y)
    dias = np.asarray(dias)
    unicos = np.unique(dias)
    filas_dia = [np.flatnonzero(dias == d) for d in unicos]
    if largo_medio is None:
        largo_medio = _largo_politis_white(y, dias)
    rng = np.random.default_rng(seed)
    nombres = [k for k in puntajes if k != referencia]
    obs = {k: metrica(y, puntajes[referencia]) - metrica(y, puntajes[k]) for k in nombres}
    reps = {k: np.empty(n_rep) for k in nombres}
    for r in range(n_rep):
        sel = np.concatenate([filas_dia[i] for i in _indices_estacionarios(len(unicos), largo_medio, rng)])
        ys = y[sel]
        m_ref = metrica(ys, puntajes[referencia][sel])
        for k in nombres:
            reps[k][r] = m_ref - metrica(ys, puntajes[k][sel])
    filas = []
    for k in nombres:
        d = reps[k]
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        filas.append({"modelo": k, "diferencia": obs[k], "ic_inf": float(np.quantile(d, 0.025)),
                      "ic_sup": float(np.quantile(d, 0.975)), "p": float(max(p, 1.0 / n_rep) if p == 0 else p),
                      "p_cero": bool(p == 0), "largo_medio": largo_medio, "n_rep": n_rep})
    return pd.DataFrame(filas)
