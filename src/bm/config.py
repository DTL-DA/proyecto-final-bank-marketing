"""Configuración global del proyecto: semilla, rutas, variables y presupuesto de los experimentos."""

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "data" / "bank-additional-full.csv"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_CORRIDAS = RUTA_RESULTADOS / "corridas"

OBJETIVO = "y"
OBJETIVO_REG = "duration"
NUMERICAS = ["age", "duration", "campaign", "pdays", "previous",
             "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
ORDEN_MESES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Decisiones heredadas del Entregable 2 (secciones 3.3.5, 3.5 y 3.7 del cuaderno)
NUM_RETIRADAS_VIF = ["euribor3m", "emp.var.rate"]
VARS_NUM = [c for c in NUMERICAS if c != "duration" and c not in NUM_RETIRADAS_VIF]
VARS_CAT = list(CATEGORICAS)
MIN_FREQ = 0.005
FRACCION_PRUEBA = 0.20

# Validación anidada
K_EXTERNO = 5
K_INTERNO = 3
PRESUPUESTO = 20  # evaluaciones (configuraciones) por optimizador en el bucle interno

METRICA_CLF = "average_precision"
METRICA_REG = "neg_root_mean_squared_error"

MODELOS_CLF = ["knn", "naive_bayes", "logistica", "arbol", "bosque", "xgboost", "svm"]
MODELOS_REG = ["knn", "ridge", "lasso", "arbol", "bosque", "xgboost", "svr"]
BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]
OPTIMIZADORES = ["grid", "random", "bayes", "genetico"]

NOMBRE_MODELO = {"knn": "KNN", "naive_bayes": "Naive Bayes", "logistica": "Logística L1/L2", "arbol": "Árbol de decisión",
                 "bosque": "Random Forest", "xgboost": "XGBoost", "svm": "SVM", "ridge": "Ridge", "lasso": "Lasso",
                 "svr": "SVR"}
NOMBRE_BALANCEO = {"ninguno": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN", "class_weight": "class_weight"}
NOMBRE_OPTIMIZADOR = {"grid": "Grid Search", "random": "Random Search", "bayes": "Bayesiana (Optuna TPE)",
                      "genetico": "Genético (DEAP)", "halving": "Successive Halving"}


def fijar_semillas(seed: int = SEED) -> None:
    """Propaga la semilla global a ``random``, ``numpy`` y a la variable de entorno de hash de Python.

    Parameters
    ----------
    seed : int
        Semilla única del proyecto. scikit-learn, XGBoost, Optuna y DEAP la reciben de forma explícita
        en cada estimador o muestreador (``random_state``/``seed``), porque no leen el estado global.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
