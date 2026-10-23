"""Componentes propios de los pipelines entrenados en la validación anidada (Bank Marketing).

Los modelos serializados en ``artifacts/`` los referencian. Conservan exactamente la lógica con la que se entrenaron,
de modo que las predicciones del dashboard coinciden con las del cuaderno (lo verifica ``tests/test_artefactos.py``).
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def pdays_a_nan(X):
    """Convierte el código 999 de ``pdays`` en faltante para imputarlo y marcarlo con un indicador."""
    X = np.asarray(X, dtype=float).copy()
    X[X == 999] = np.nan
    return X


class KNNBalanceado(KNeighborsClassifier):
    """KNN con votos ponderados por el inverso de la frecuencia de cada clase.

    ``KNeighborsClassifier`` no admite ``class_weight``. Ponderar el voto de cada vecino de la clase *c* por
    ``n / (k · n_c)`` es lo que haría ``class_weight='balanced'`` si el clasificador aceptara pesos por muestra.
    """

    def fit(self, X, y):
        super().fit(X, y)
        _, conteos = np.unique(y, return_counts=True)
        self.pesos_clase_ = len(y) / (len(conteos) * conteos)
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X) * self.pesos_clase_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class XGBBalanceado(XGBClassifier):
    """XGBoost con ``scale_pos_weight = n_negativos / n_positivos`` calculado en cada ajuste."""

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kwargs)
