"""Preprocesamiento encapsulado: el mismo ``ColumnTransformer`` del Entregable 2 (sección 3.7 del cuaderno)."""

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as cfg

# En validación cronológica aparecen meses no vistos en el ajuste; el codificador los envía a la categoría
# infrecuente (o a la de referencia) por diseño, así que el aviso se silencia.
warnings.filterwarnings("ignore", message="Found unknown categories")


def pdays_a_nan(X):
    """Convierte el código 999 de ``pdays`` en faltante para imputarlo y marcarlo con un indicador."""
    X = np.asarray(X, dtype=float).copy()
    X[X == 999] = np.nan
    return X


def construir_preprocesador(numericas=None, categoricas=None) -> ColumnTransformer:
    """Construye el preprocesador sin ajustar.

    Parameters
    ----------
    numericas, categoricas : list of str, optional
        Variables de entrada; por defecto ``VARS_NUM`` (sin ``duration`` ni las retiradas por VIF) y ``VARS_CAT``.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Estandariza las numéricas, aplica ``log1p`` a ``campaign`` y ``previous``, trata ``pdays = 999`` como faltante
        con indicador e imputación por la mediana, y codifica las categóricas con *one-hot* (referencia ``drop='first'``
        y categorías con menos del 0,5 % agrupadas como infrecuentes). Todos los parámetros se aprenden en ``fit``,
        así que dentro de un ``Pipeline`` solo usan los datos de entrenamiento de cada pliegue.
    """
    numericas = cfg.VARS_NUM if numericas is None else numericas
    categoricas = cfg.VARS_CAT if categoricas is None else categoricas
    escala = [c for c in numericas if c not in ("campaign", "previous", "pdays")]
    logs = [c for c in ("campaign", "previous") if c in numericas]
    pasos = [
        ("num", StandardScaler(), escala),
        ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), logs),
        ("cat", OneHotEncoder(drop="first", min_frequency=cfg.MIN_FREQ, handle_unknown="infrequent_if_exist",
                              sparse_output=False), categoricas),
    ]
    if "pdays" in numericas:
        pasos.insert(2, ("pdays", make_pipeline(
            FunctionTransformer(pdays_a_nan, feature_names_out="one-to-one"),
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler()), ["pdays"]))
    return ColumnTransformer(pasos)
