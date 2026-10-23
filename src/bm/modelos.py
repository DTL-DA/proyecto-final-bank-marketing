"""Definición de los modelos base y de las estrategias de balanceo como pipelines de ``imblearn``."""

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config as cfg
from .preprocesamiento import construir_preprocesador


class KNNBalanceado(KNeighborsClassifier):
    """KNN con votos ponderados por el inverso de la frecuencia de cada clase.

    ``KNeighborsClassifier`` no admite ``class_weight``. Ponderar el voto de cada vecino de la clase *c* por
    ``n / (k · n_c)`` es exactamente lo que haría ``class_weight='balanced'`` si el clasificador aceptara pesos por
    muestra: la proporción de vecinos de cada clase se reescala y se vuelve a normalizar.
    """

    def fit(self, X, y):
        """Ajusta el KNN y guarda los pesos de clase, n / (n_clases · n_c), calculados con los datos de entrenamiento."""
        super().fit(X, y)
        _, conteos = np.unique(y, return_counts=True)
        self.pesos_clase_ = len(y) / (len(conteos) * conteos)
        return self

    def predict_proba(self, X):
        """Proporción de vecinos de cada clase reescalada por los pesos de clase y normalizada para que sume 1."""
        p = super().predict_proba(X) * self.pesos_clase_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Clase con la mayor probabilidad reponderada."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class XGBBalanceado(XGBClassifier):
    """XGBoost con ``scale_pos_weight = n_negativos / n_positivos`` calculado en cada ajuste.

    Es el equivalente de ``class_weight='balanced'`` en XGBoost para el caso binario y se calcula con los datos de
    entrenamiento de cada pliegue, nunca con los de validación.
    """

    def fit(self, X, y, **kwargs):
        """Calcula ``scale_pos_weight`` con las clases de este ajuste y entrena XGBoost."""
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kwargs)


def _estimador_clf(modelo: str, balanceado: bool, seed: int):
    cw = "balanced" if balanceado else None
    if modelo == "knn":
        return [("modelo", (KNNBalanceado if balanceado else KNeighborsClassifier)(algorithm="brute"))]
    if modelo == "naive_bayes":
        # Con pesos balanceados, las medias y varianzas condicionales no cambian y las priors se igualan:
        # class_weight='balanced' en GaussianNB equivale exactamente a priors = [0,5; 0,5].
        return [("modelo", GaussianNB(priors=[0.5, 0.5] if balanceado else None))]
    if modelo == "logistica":
        return [("modelo", LogisticRegression(solver="saga", max_iter=1000, tol=1e-3, class_weight=cw,
                                              random_state=seed))]
    if modelo == "arbol":
        return [("modelo", DecisionTreeClassifier(class_weight=cw, random_state=seed))]
    if modelo == "bosque":
        return [("modelo", RandomForestClassifier(class_weight=cw, n_jobs=1, random_state=seed))]
    if modelo == "xgboost":
        clase = XGBBalanceado if balanceado else XGBClassifier
        return [("modelo", clase(tree_method="hist", colsample_bytree=0.8, eval_metric="logloss", n_jobs=1,
                                 random_state=seed, verbosity=0))]
    if modelo == "svm":
        return [("nys", Nystroem(kernel="rbf", random_state=seed)),
                ("modelo", LinearSVC(class_weight=cw, max_iter=2000, random_state=seed))]
    raise ValueError(f"Modelo de clasificación desconocido: {modelo}")


def _estimador_reg(modelo: str, seed: int):
    if modelo == "knn":
        return [("modelo", KNeighborsRegressor(algorithm="brute"))]
    if modelo == "ridge":
        return [("modelo", Ridge(random_state=seed))]
    if modelo == "lasso":
        return [("modelo", Lasso(max_iter=5000, random_state=seed))]
    if modelo == "arbol":
        return [("modelo", DecisionTreeRegressor(random_state=seed))]
    if modelo == "bosque":
        return [("modelo", RandomForestRegressor(n_jobs=1, random_state=seed))]
    if modelo == "xgboost":
        return [("modelo", XGBRegressor(tree_method="hist", colsample_bytree=0.8, n_jobs=1, random_state=seed,
                                        verbosity=0))]
    if modelo == "svr":
        return [("nys", Nystroem(kernel="rbf", n_components=300, random_state=seed)),
                # LinearSVR penaliza el intercepto; con el objetivo estandarizado el intercepto queda cerca de cero
                ("modelo", TransformedTargetRegressor(
                    regressor=LinearSVR(max_iter=5000, dual="auto", random_state=seed), transformer=StandardScaler()))]
    raise ValueError(f"Modelo de regresión desconocido: {modelo}")


def construir_pipeline(tarea: str, modelo: str, balanceo: str = "ninguno", seed: int = cfg.SEED) -> Pipeline:
    """Arma el pipeline completo preprocesamiento → (balanceo) → modelo.

    Parameters
    ----------
    tarea : {"clf", "reg"}
        Clasificación de ``y`` o regresión de ``log1p(duration)``.
    modelo : str
        Clave del modelo (``config.MODELOS_CLF`` o ``config.MODELOS_REG``).
    balanceo : {"ninguno", "smote", "adasyn", "class_weight"}
        Solo aplica en clasificación. SMOTE y ADASYN van dentro del ``Pipeline`` de ``imblearn``, de modo que el
        sobremuestreo ocurre únicamente en ``fit`` (datos de entrenamiento del pliegue) y nunca al predecir.
    seed : int
        Semilla propagada a todos los componentes aleatorios.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline sin ajustar, con pasos ``pre``, ``bal`` (opcional), ``nys`` (solo SVM/SVR) y ``modelo``.
    """
    pasos = [("pre", construir_preprocesador())]
    if tarea == "reg":
        return Pipeline(pasos + _estimador_reg(modelo, seed))
    if balanceo == "smote":
        pasos.append(("bal", SMOTE(random_state=seed)))
    elif balanceo == "adasyn":
        pasos.append(("bal", ADASYN(random_state=seed)))
    return Pipeline(pasos + _estimador_clf(modelo, balanceo == "class_weight", seed))
