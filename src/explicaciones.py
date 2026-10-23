"""Valores SHAP por muestreo de permutaciones, sin cargar la librería ``shap`` en el servidor.

El plan gratuito de Render tiene poca memoria y la librería ``shap`` (con ``numba``) no cabe junto con los siete
modelos. Este módulo implementa el mismo estimador que el ``PermutationExplainer`` de ``shap``: valores de Shapley
con función de valor intervencional, estimados con permutaciones antitéticas (Štrumbelj y Kononenko, 2014).
"""
import numpy as np
import pandas as pd


def shap_permutaciones(f, x, fondo, variables, n_perm=64, semilla=0):
    """Estima los valores SHAP de una observación sobre las variables originales (antes del *one-hot*).

    Parameters
    ----------
    f : callable
        Recibe un ``DataFrame`` con las columnas de entrada del modelo y devuelve el puntaje de cada fila
        (por ejemplo, la probabilidad de suscribir).
    x : pandas.DataFrame
        Una fila con la observación que se explica.
    fondo : pandas.DataFrame
        Observaciones de referencia; definen el valor esperado contra el que se mide cada contribución.
    variables : list of str
        Variables que se explican. Las columnas que no están en la lista conservan el valor de ``x``.
    n_perm : int, default 64
        Número de permutaciones; se usan en pares antitéticos (π y su inversa), así que se redondea a par.
    semilla : int, default 0
        Semilla del muestreo de permutaciones y de filas de referencia.

    Returns
    -------
    phi : numpy.ndarray
        Contribución de cada variable, en las unidades de ``f``.
    base : float
        Promedio de ``f`` sobre las filas de referencia muestreadas (valor esperado).
    fx : float
        Puntaje de la observación, ``f(x)``.

    Notes
    -----
    Para cada permutación π y una fila de referencia z se parte de z y se cambian, en el orden de π, sus valores por
    los de x; la diferencia de ``f`` en cada paso es la contribución marginal de la variable cambiada. Por
    construcción se cumple la propiedad de eficiencia: ``base + phi.sum() == fx``.
    """
    rng = np.random.default_rng(semilla)
    p = len(variables)
    m = max(2, n_perm + n_perm % 2)
    perms = []
    for _ in range(m // 2):
        pi = rng.permutation(p)
        perms += [pi, pi[::-1]]
    perms = np.array(perms)
    idx_z = rng.integers(0, len(fondo), size=m // 2).repeat(2)

    # posicion[b, j]: paso en que la variable j pasa de z a x en la permutación b
    posicion = np.empty_like(perms)
    posicion[np.arange(m)[:, None], perms] = np.arange(p)[None, :]
    aun_de_z = (posicion[:, None, :] >= np.arange(p + 1)[None, :, None]).reshape(-1, p)

    z = fondo.iloc[idx_z][variables].to_numpy(dtype=object)
    valores = np.where(aun_de_z, np.repeat(z, p + 1, axis=0), x[variables].to_numpy(dtype=object)[0][None, :])
    lote = pd.DataFrame(np.repeat(x.to_numpy(dtype=object), m * (p + 1), axis=0), columns=x.columns)
    lote[variables] = valores
    lote = lote.astype(x.dtypes.to_dict())

    y = np.asarray(f(lote), dtype=float).reshape(m, p + 1)
    contribuciones = np.diff(y, axis=1)
    phi = np.zeros(p)
    np.add.at(phi, perms.ravel(), contribuciones.ravel())
    return phi / m, float(y[:, 0].mean()), float(y[0, -1])
