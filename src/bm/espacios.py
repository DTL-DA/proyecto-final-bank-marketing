"""Espacios de búsqueda de hiperparámetros, compartidos por todos los optimizadores.

Cada hiperparámetro se describe con un diccionario:

- ``{"tipo": "float", "bajo": a, "alto": b, "log": bool}``
- ``{"tipo": "int", "bajo": a, "alto": b, "log": bool}``
- ``{"tipo": "cat", "opciones": [...]}``

Grid Search usa la rejilla explícita ``REJILLAS`` (a lo sumo ``PRESUPUESTO`` puntos); Random Search, Optuna y DEAP
muestrean el espacio continuo ``ESPACIOS``. Así, todos los métodos buscan en la misma región y con el mismo
presupuesto de evaluaciones.
"""

import itertools

import numpy as np

F = lambda a, b, log=False: {"tipo": "float", "bajo": a, "alto": b, "log": log}  # noqa: E731
I = lambda a, b, log=False: {"tipo": "int", "bajo": a, "alto": b, "log": log}  # noqa: E731
C = lambda *ops: {"tipo": "cat", "opciones": list(ops)}  # noqa: E731

_KNN = {"modelo__n_neighbors": I(5, 300, log=True), "modelo__weights": C("uniform", "distance"),
        "modelo__p": C(1, 2)}
_KNN_GRID = {"modelo__n_neighbors": [5, 15, 40, 100, 300], "modelo__weights": ["uniform", "distance"],
             "modelo__p": [1, 2]}
_BOSQUE_GRID = {"modelo__n_estimators": [100, 250], "modelo__max_depth": [8, 20],
                "modelo__min_samples_leaf": [1, 20], "modelo__max_features": ["sqrt", 0.5]}
_XGB = {"modelo__n_estimators": I(50, 500), "modelo__learning_rate": F(0.01, 0.3, log=True),
        "modelo__max_depth": I(2, 10), "modelo__subsample": F(0.5, 1.0)}
_XGB_GRID = {"modelo__n_estimators": [100, 300], "modelo__learning_rate": [0.03, 0.1],
             "modelo__max_depth": [3, 6], "modelo__subsample": [0.7, 1.0]}

ESPACIOS = {
    ("clf", "knn"): _KNN,
    ("clf", "naive_bayes"): {"modelo__var_smoothing": F(1e-12, 1.0, log=True)},
    ("clf", "logistica"): {"modelo__C": F(1e-4, 1e2, log=True), "modelo__l1_ratio": C(0.0, 1.0)},
    ("clf", "arbol"): {"modelo__max_depth": I(2, 30), "modelo__min_samples_leaf": I(1, 1000, log=True),
                       "modelo__criterion": C("gini", "entropy")},
    ("clf", "bosque"): {"modelo__n_estimators": I(50, 300), "modelo__max_depth": I(3, 30),
                        "modelo__min_samples_leaf": I(1, 200, log=True),
                        "modelo__max_features": C("sqrt", "log2", 0.5)},
    ("clf", "xgboost"): _XGB,
    ("clf", "svm"): {"nys__gamma": F(1e-3, 1.0, log=True), "nys__n_components": I(100, 600),
                     "modelo__C": F(1e-3, 10.0, log=True)},
    ("reg", "knn"): _KNN,
    ("reg", "ridge"): {"modelo__alpha": F(1e-3, 1e4, log=True)},
    ("reg", "lasso"): {"modelo__alpha": F(1e-5, 1.0, log=True)},
    ("reg", "arbol"): {"modelo__max_depth": I(2, 30), "modelo__min_samples_leaf": I(1, 1000, log=True),
                       "modelo__max_features": C(1.0, 0.5, "sqrt")},
    ("reg", "bosque"): {"modelo__n_estimators": I(50, 300), "modelo__max_depth": I(3, 30),
                        "modelo__min_samples_leaf": I(1, 200, log=True),
                        "modelo__max_features": C("sqrt", 0.33, 0.5)},
    ("reg", "xgboost"): _XGB,
    ("reg", "svr"): {"nys__gamma": F(1e-3, 1.0, log=True), "modelo__regressor__C": F(1e-3, 10.0, log=True),
                     "modelo__regressor__epsilon": F(0.0, 0.5)},
}

REJILLAS = {
    ("clf", "knn"): _KNN_GRID,
    ("clf", "naive_bayes"): {"modelo__var_smoothing": list(np.logspace(-12, 0, 20))},
    ("clf", "logistica"): {"modelo__C": list(np.logspace(-4, 2, 10)), "modelo__l1_ratio": [0.0, 1.0]},
    ("clf", "arbol"): {"modelo__max_depth": [3, 5, 8, 12, 20], "modelo__min_samples_leaf": [5, 50],
                       "modelo__criterion": ["gini", "entropy"]},
    ("clf", "bosque"): _BOSQUE_GRID,
    ("clf", "xgboost"): _XGB_GRID,
    ("clf", "svm"): {"nys__gamma": [0.003, 0.03, 0.3], "nys__n_components": [200, 500],
                     "modelo__C": [0.01, 0.1, 1.0]},
    ("reg", "knn"): _KNN_GRID,
    ("reg", "ridge"): {"modelo__alpha": list(np.logspace(-3, 4, 20))},
    ("reg", "lasso"): {"modelo__alpha": list(np.logspace(-5, 0, 20))},
    ("reg", "arbol"): {"modelo__max_depth": [3, 5, 8, 12, 20], "modelo__min_samples_leaf": [5, 50],
                       "modelo__max_features": [1.0, 0.5]},
    ("reg", "bosque"): {**_BOSQUE_GRID, "modelo__max_features": [0.33, 0.5]},
    ("reg", "xgboost"): _XGB_GRID,
    ("reg", "svr"): {"nys__gamma": [0.003, 0.03, 0.3], "modelo__regressor__C": [0.01, 0.1, 1.0],
                     "modelo__regressor__epsilon": [0.05, 0.3]},
}

JUSTIFICACION = {
    "knn": "k de 5 a 300 en escala log: con miles de registros por pliegue, k pequeño sobreajusta y k cercano a 300 "
           "suaviza hacia la prevalencia; ponderación uniforme o por distancia y métrica Manhattan (p = 1) o "
           "euclídea (p = 2), relevante con decenas de columnas binarias.",
    "naive_bayes": "var_smoothing de 1e-12 a 1 (log): las columnas one-hot tienen varianzas casi nulas dentro de "
                   "una clase y el suavizado controla esa inestabilidad; es el único hiperparámetro continuo.",
    "logistica": "C de 1e-4 a 100 (log), el rango donde la logística pasa de casi constante a casi sin penalizar "
                 "(en el Entregable 2 el óptimo por log-loss fue 0,01); penalización L1 o L2 vía l1_ratio.",
    "arbol": "Profundidad de 2 a 30 y hojas de 1 a 1000 registros (log): controlan el sesgo-varianza del árbol, "
             "que sin restricciones memoriza el entrenamiento; criterio o fracción de variables por división.",
    "bosque": "50 a 300 árboles (más árboles no cambian el sesgo y encarecen), profundidad 3 a 30, hojas 1 a 200 "
              "(log) y fracción de variables por división, el principal mecanismo de decorrelación.",
    "xgboost": "Tasa de aprendizaje 0,01 a 0,3 (log) y número de árboles 50 a 500, que interactúan (menor tasa, "
               "más árboles); profundidad 2 a 10 y submuestreo 0,5 a 1 como regularización estocástica.",
    "svm": "Núcleo RBF aproximado con Nyström (100 a 600 componentes) y SVM lineal: gamma de 1e-3 a 1 y C de 1e-3 a "
           "10 en escala log, con la interacción C × gamma típica del núcleo gaussiano.",
    "ridge": "alpha de 1e-3 a 1e4 (log): desde casi mínimos cuadrados hasta una contracción fuerte.",
    "lasso": "alpha de 1e-5 a 1 (log): por encima de 1 el Lasso anula todos los coeficientes con el objetivo "
             "estandarizado en escala log.",
    "svr": "Nyström RBF (300 componentes) con SVR lineal sobre el objetivo estandarizado: gamma y C en escala log y "
           "epsilon de 0 a 0,5 desviaciones estándar del objetivo.",
}


def n_puntos_rejilla(tarea: str, modelo: str) -> int:
    """Número de configuraciones de la rejilla de Grid Search."""
    return int(np.prod([len(v) for v in REJILLAS[(tarea, modelo)].values()]))


def configuraciones_rejilla(tarea: str, modelo: str) -> list[dict]:
    """Producto cartesiano de la rejilla en orden determinista."""
    rej = REJILLAS[(tarea, modelo)]
    return [dict(zip(rej, valores)) for valores in itertools.product(*rej.values())]


def decodificar(espacio: dict, u: np.ndarray) -> dict:
    """Transforma un vector en [0, 1]^d en una configuración válida del espacio.

    Parameters
    ----------
    espacio : dict
        Espacio de ``ESPACIOS``.
    u : numpy.ndarray
        Coordenadas uniformes, una por hiperparámetro y en el orden del diccionario.

    Returns
    -------
    dict
        Configuración con enteros redondeados, escalas logarítmicas respetadas y categorías elegidas por tramos iguales.
    """
    params = {}
    for (nombre, e), ui in zip(espacio.items(), np.clip(u, 0.0, 1.0)):
        if e["tipo"] == "cat":
            params[nombre] = e["opciones"][min(int(ui * len(e["opciones"])), len(e["opciones"]) - 1)]
            continue
        if e["log"]:
            v = float(np.exp(np.log(e["bajo"]) + ui * (np.log(e["alto"]) - np.log(e["bajo"]))))
        else:
            v = float(e["bajo"] + ui * (e["alto"] - e["bajo"]))
        params[nombre] = int(round(v)) if e["tipo"] == "int" else v
    return params
