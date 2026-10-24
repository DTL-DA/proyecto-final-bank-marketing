"""Pruebas unitarias del modelo original (L2E) sobre la copia de producción del dashboard, src/original_model.py.

Son las mismas del cuaderno del cuarto entregable (cuarto_entregable/tests/test_modelo_original.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.original_model import LogisticaDosEscalas, filtrar_nivel  # noqa: E402

PRODUCCION = Path(__file__).resolve().parents[1] / "src" / "original_model.py"
INVESTIGACION = Path(__file__).resolve().parents[1] / "cuarto_entregable" / "src" / "bm" / "modelo_original.py"


@pytest.mark.skipif(not INVESTIGACION.exists(), reason="el paquete de investigación no está en esta copia")
def test_algoritmo_identico_al_del_cuaderno():
    """La copia de producción ejecuta exactamente el mismo código que el modelo evaluado en el cuaderno."""
    def codigo(ruta):
        return ruta.read_text(encoding="utf-8").split('"""\n\nfrom collections import deque', 1)[1]

    assert codigo(PRODUCCION) == codigo(INVESTIGACION)

BETA = np.array([1.0, -0.5, 0.25])


def simular(dias=60, por_dia=150, q=0.05, seed=0, inicio=-1.0):
    """Datos con solución conocida: nivel en paseo aleatorio de varianza q y pendientes fijas BETA."""
    rng = np.random.default_rng(seed)
    u = inicio + np.cumsum(rng.normal(0, np.sqrt(q), dias))
    t = np.repeat(np.arange(dias), por_dia)
    Xf = rng.normal(size=(len(t), len(BETA)))
    y = rng.binomial(1, expit(u[t] + Xf @ BETA))
    return np.column_stack([Xf, t]), y, u


@pytest.fixture(scope="module")
def datos():
    return simular()


@pytest.fixture(scope="module")
def ajustado(datos):
    X, y, _ = datos
    return LogisticaDosEscalas(lam=1.0, q_nivel=0.05, q_pendiente=1e-6).fit(X, y)


# -------------------------------------------------------------- (i) aprende cuando la solución es conocida

def test_recupera_pendientes_y_nivel(datos, ajustado):
    _, _, u = datos
    assert np.max(np.abs(ajustado.coef_ - BETA)) < 0.15
    assert np.corrcoef(ajustado.nivel_, u)[0, 1] > 0.9


def test_filtro_supera_al_modelo_estatico_cuando_el_nivel_deriva():
    X, y, _ = simular(dias=120, q=0.08, seed=3)
    corte = (X[:, -1] < 60)
    m = LogisticaDosEscalas(lam=1.0, q_nivel=0.08, q_pendiente=1e-6).fit(X[corte], y[corte])
    p_sec = m.predict_proba_secuencial(X[~corte], y[~corte])[:, 1]
    p_est = m.predict_proba(X[~corte])[:, 1]
    assert roc_auc_score(y[~corte], p_sec) > roc_auc_score(y[~corte], p_est)
    assert log_loss(y[~corte], p_sec) < log_loss(y[~corte], p_est) - 0.02


# ------------------------------------------------------------------------------- (ii) reproducibilidad

def test_ajuste_determinista(datos):
    X, y, _ = datos
    a = LogisticaDosEscalas(lam=2.0, q_nivel=0.1).fit(X, y)
    b = LogisticaDosEscalas(lam=2.0, q_nivel=0.1).fit(X, y)
    assert np.array_equal(a.coef_, b.coef_) and np.array_equal(a.nivel_, b.nivel_)
    assert np.array_equal(a.predict_proba_secuencial(X, y), b.predict_proba_secuencial(X, y))


# ------------------------------------------------------------------------------ (iii) entradas inválidas

@pytest.mark.parametrize("parametros", [{"lam": 0.0}, {"lam": -1.0}, {"q_nivel": -0.1}, {"q_pendiente": np.nan},
                                        {"q_nivel": 0.0}, {"retardo": 0}, {"retardo": 1.5}, {"max_iter": 0}])
def test_hiperparametros_invalidos(datos, parametros):
    X, y, _ = datos
    with pytest.raises(ValueError):
        LogisticaDosEscalas(**parametros).fit(X, y)


def test_datos_invalidos(datos):
    X, y, _ = datos
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        LogisticaDosEscalas().fit(X_nan, y)
    with pytest.raises(ValueError):
        LogisticaDosEscalas().fit(X, np.zeros_like(y))  # una sola clase
    X_t = X.copy()
    X_t[:, -1] += 0.5  # día no entero
    with pytest.raises(ValueError):
        LogisticaDosEscalas().fit(X_t, y)
    with pytest.raises(ValueError):
        LogisticaDosEscalas().fit(X[:, -1:], y)  # sin predictoras
    with pytest.raises(NotFittedError):
        LogisticaDosEscalas().predict_proba(X)


def test_prediccion_invalida(ajustado, datos):
    X, y, _ = datos
    with pytest.raises(ValueError):
        ajustado.predict_proba(X[:, 1:])  # número de columnas distinto
    with pytest.raises(ValueError):
        ajustado.predict_proba_secuencial(X, y[:-1])  # longitudes distintas
    with pytest.raises(ValueError):
        ajustado.predict_proba_secuencial(X, np.full_like(y, 7))  # etiqueta no vista
    with pytest.raises(ValueError):
        ajustado.predict_proba_secuencial(X, y, retardo=0)


# ------------------------------------------------------------------------- (iv) compatibilidad scikit-learn

def test_clone_y_parametros(ajustado):
    c = clone(ajustado)
    assert c.get_params() == ajustado.get_params()
    assert not hasattr(c, "coef_")
    c.set_params(lam=5.0, q_pendiente=1e-3)
    assert c.lam == 5.0 and c.q_pendiente == 1e-3


def test_integra_con_validacion_cruzada(datos):
    X, y, _ = datos
    puntajes = cross_val_score(LogisticaDosEscalas(), X, y, cv=TimeSeriesSplit(3), scoring="roc_auc")
    assert np.all(np.isfinite(puntajes)) and puntajes.mean() > 0.6


def test_salidas_bien_formadas(ajustado, datos):
    X, y, _ = datos
    p = ajustado.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1.0) and np.all((p >= 0) & (p <= 1))
    assert set(np.unique(ajustado.predict(X))) <= set(ajustado.classes_)


# --------------------------------------------------------------------- propiedades del algoritmo

def test_sin_fuga_de_etiquetas(ajustado):
    X, y, _ = simular(dias=20, seed=7)
    X[:, -1] += 100  # días posteriores al ajuste
    base = ajustado.predict_proba_secuencial(X, y)[:, 1]
    dias = X[:, -1]
    k = 110
    y_mod = y.copy()
    y_mod[dias >= k] = 1 - y_mod[dias >= k]  # se alteran las etiquetas desde el día k
    alterado = ajustado.predict_proba_secuencial(X, y_mod)[:, 1]
    assert np.array_equal(base[dias <= k], alterado[dias <= k])  # con retardo 1, el día k no ve su propia etiqueta
    assert not np.allclose(base[dias == k + 1], alterado[dias == k + 1])
    y_r = y.copy()
    y_r[dias == k] = 1 - y_r[dias == k]
    con_retardo = ajustado.predict_proba_secuencial(X, y_r, retardo=3)[:, 1]
    referencia = ajustado.predict_proba_secuencial(X, y, retardo=3)[:, 1]
    assert np.array_equal(con_retardo[dias <= k + 2], referencia[dias <= k + 2])


def test_objetivo_no_decrece_y_converge(ajustado):
    assert ajustado.convergio_
    assert np.all(np.diff(ajustado.objetivo_) >= -1e-9)


def test_caso_limite_equivale_a_logistica_l2(datos):
    X, y, _ = datos
    lam = 3.0
    m = LogisticaDosEscalas(lam=lam, nivel_latente=False, q_nivel=0.0, tol=1e-12).fit(X, y)
    ref = LogisticRegression(C=1 / lam, tol=1e-10, max_iter=10_000).fit(X[:, :-1], y)
    assert np.allclose(m.coef_, ref.coef_.ravel(), atol=1e-4)
    assert np.isclose(m.estado_media_[0], ref.intercept_[0], atol=1e-4)


def test_sin_actualizacion_igual_a_prediccion_estatica(datos):
    X, y, _ = datos
    m = LogisticaDosEscalas(actualizar_nivel=False, actualizar_pendientes=False).fit(X, y)
    assert np.allclose(m.predict_proba_secuencial(X, y), m.predict_proba(X))


def test_actualizar_equivale_a_la_prediccion_secuencial(ajustado):
    X, y, _ = simular(dias=6, seed=11)
    X[:, -1] += 100
    secuencial = ajustado.predict_proba_secuencial(X, y)[:, 1]
    m = clone(ajustado).fit(*simular()[:2])
    dias = X[:, -1]
    m.actualizar(X[dias < 105], y[dias < 105])
    assert np.allclose(m.predict_proba(X[dias == 105])[:, 1], secuencial[dias == 105])


def test_filtro_aislado_sigue_el_nivel():
    rng = np.random.default_rng(5)
    dias = np.repeat(np.arange(80), 100)
    nivel = np.r_[np.full(40, -2.0), np.full(40, 0.5)][dias]  # salto de régimen
    y = rng.binomial(1, expit(nivel))
    p = filtrar_nivel(np.full(len(y), 0.5), dias, y, q=0.1)
    assert abs(p[dias >= 45].mean() - expit(0.5)) < 0.05 and abs(p[(dias > 5) & (dias < 40)].mean() - expit(-2)) < 0.05


# ----------------------------------------------------------------- mejoras M2 y M3 (sección de mejoras)

def test_semivida_por_defecto_no_cambia_el_ajuste(datos):
    X, y, _ = datos
    a = LogisticaDosEscalas(lam=2.0, q_nivel=0.1).fit(X, y)
    b = LogisticaDosEscalas(lam=2.0, q_nivel=0.1, semivida=None).fit(X, y)
    c = LogisticaDosEscalas(lam=2.0, q_nivel=0.1, semivida=1e9).fit(X, y)
    assert np.array_equal(a.coef_, b.coef_)
    assert np.allclose(a.coef_, c.coef_, atol=1e-6)
    with pytest.raises(ValueError):
        LogisticaDosEscalas(semivida=0.0).fit(X, y)


def test_semivida_da_mas_peso_al_final():
    rng = np.random.default_rng(2)
    t = np.repeat(np.arange(80), 120)
    x = rng.normal(size=len(t))
    beta = np.where(t < 40, -1.0, 1.0)  # la pendiente cambia de signo a mitad del periodo
    y = rng.binomial(1, expit(-0.5 + beta * x))
    X = np.column_stack([x, t])
    sin = LogisticaDosEscalas(lam=1.0, q_nivel=0.01).fit(X, y)
    con = LogisticaDosEscalas(lam=1.0, q_nivel=0.01, semivida=10).fit(X, y)
    assert con.coef_[0] > sin.coef_[0] and con.coef_[0] > 0.5


def test_bayes_empirico_recupera_el_orden_de_q():
    X, y, _ = simular(dias=150, por_dia=200, q=0.05, seed=4)
    m = LogisticaDosEscalas(lam=1.0, q_nivel_empirico=True).fit(X, y)
    assert 0.01 < m.q_nivel_ < 0.25
    ev = []
    for q in (m.q_nivel_ / 10, m.q_nivel_, m.q_nivel_ * 10):
        mm = LogisticaDosEscalas(lam=1.0, q_nivel=q).fit(X, y)
        t, F = mm._separar(np.asarray(X, dtype=float))
        dias, idx = np.unique(t, return_inverse=True)
        datos = (F, idx, y.astype(float), np.ones(len(y)), np.diff(dias).astype(float))
        u, b, prec, _ = mm._newton(datos, 1.0, q)
        ev.append(mm._log_evidencia(datos, 1.0, u, b, prec))
    assert ev[1] >= ev[0] and ev[1] >= ev[2]
