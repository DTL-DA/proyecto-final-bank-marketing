"""Estimador SHAP por permutaciones: exactitud en un caso con solución conocida, eficiencia y reproducibilidad."""
import numpy as np
import pandas as pd

from src.data import DATOS, fila_entrada, fondo, probabilidad
from src.explicaciones import shap_permutaciones


def test_modelo_lineal_tiene_solucion_exacta():
    """Para f lineal y una sola fila de referencia z, el valor de Shapley es w_j · (x_j − z_j)."""
    w = np.array([2.0, -1.0, 0.5])

    def f(D):
        return D[["a", "b", "c"]].to_numpy(dtype=float) @ w

    x = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [-2.0]})
    z = pd.DataFrame({"a": [0.0], "b": [1.0], "c": [4.0]})
    phi, base, fx = shap_permutaciones(f, x, z, ["a", "b", "c"], n_perm=8)
    np.testing.assert_allclose(phi, w * (x.iloc[0].to_numpy() - z.iloc[0].to_numpy()))
    assert np.isclose(base + phi.sum(), fx)


def test_eficiencia_y_reproducibilidad_con_un_modelo_real():
    import json

    esquema = json.loads((DATOS / "esquema.json").read_text(encoding="utf-8"))
    variables = [v["variable"] for v in esquema["variables"]]
    x = fila_entrada({v["variable"]: v["defecto"] for v in esquema["variables"]})

    def f(D):
        return probabilidad("naive_bayes", D)

    phi1, base1, fx1 = shap_permutaciones(f, x, fondo(), variables, n_perm=32, semilla=3)
    phi2, base2, fx2 = shap_permutaciones(f, x, fondo(), variables, n_perm=32, semilla=3)
    np.testing.assert_array_equal(phi1, phi2)
    assert np.isclose(base1 + phi1.sum(), fx1)
    assert np.isclose(fx1, f(x)[0])
