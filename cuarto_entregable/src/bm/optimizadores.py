"""Optimizadores de hiperparámetros sobre un evaluador común con presupuesto, caché y registro *anytime*.

Todos los métodos llaman al mismo ``Evaluador``, que ajusta el ``Pipeline`` completo (preprocesamiento, balanceo y
modelo) dentro de cada pliegue de la validación interna. Así, ningún método ve datos de validación al preprocesar o
balancear, el presupuesto se mide igual para todos (configuraciones distintas evaluadas) y las curvas de desempeño
en cualquier momento son comparables.
"""

import math
import random
import time
import warnings

import numpy as np
import optuna
from deap import base, creator, tools
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate

from . import config as cfg
from .espacios import ESPACIOS, configuraciones_rejilla, decodificar

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _clave(params: dict) -> tuple:
    norm = []
    for k, v in sorted(params.items()):
        if isinstance(v, (float, np.floating)):
            v = float(f"{float(v):.10g}")
        elif isinstance(v, np.integer):
            v = int(v)
        norm.append((k, v))
    return tuple(norm)


class Evaluador:
    """Evalúa configuraciones con validación cruzada interna, respetando un presupuesto.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        Pipeline sin ajustar (preprocesamiento + balanceo + modelo).
    X, y : array-like
        Datos de entrenamiento del pliegue externo.
    cv : splitter
        Validación interna (``TimeSeriesSplit``).
    scoring : str
        Métrica a maximizar (``average_precision`` o ``neg_root_mean_squared_error``).
    presupuesto : int
        Número máximo de configuraciones distintas a evaluar. Las repetidas salen del caché y no consumen presupuesto.

    Notes
    -----
    Cada evaluación nueva queda en ``historial`` con su puntaje medio, los puntajes por pliegue, su tiempo de pared y
    el tiempo acumulado desde que empezó la optimización (incluye el costo propio del optimizador).
    """

    def __init__(self, pipeline, X, y, cv, scoring, presupuesto=cfg.PRESUPUESTO):
        self.pipeline, self.X, self.y, self.cv, self.scoring = pipeline, X, y, cv, scoring
        self.presupuesto = presupuesto
        self.cache, self.historial = {}, []
        self.t0 = time.perf_counter()

    @property
    def restantes(self) -> int:
        """Configuraciones nuevas que aún caben en el presupuesto."""
        return self.presupuesto - len(self.historial)

    def evaluar(self, params: dict) -> tuple[float, np.ndarray]:
        """Devuelve (puntaje medio, puntajes por pliegue); ``-inf`` si la configuración falla o no hay presupuesto."""
        clave = _clave(params)
        if clave in self.cache:
            return self.cache[clave]
        if self.restantes <= 0:
            return -np.inf, np.array([np.nan])
        t = time.perf_counter()
        modelo = clone(self.pipeline).set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = cross_validate(modelo, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=1, error_score=np.nan)
        pliegues = np.asarray(r["test_score"], dtype=float)
        media = float(np.mean(pliegues)) if np.all(np.isfinite(pliegues)) else -np.inf
        self.cache[clave] = (media, pliegues)
        mejor = max([h["puntaje"] for h in self.historial] + [media])
        self.historial.append({"n": len(self.historial) + 1, "params": dict(params), "puntaje": media,
                               "pliegues": pliegues.tolist(), "t_eval": time.perf_counter() - t,
                               "t_acum": time.perf_counter() - self.t0, "mejor": mejor})
        return self.cache[clave]

    def __call__(self, params: dict) -> float:
        return self.evaluar(params)[0]

    def mejor(self) -> tuple[dict, float]:
        """Mejor configuración observada y su puntaje medio de validación interna."""
        h = max(self.historial, key=lambda r: r["puntaje"])
        return h["params"], h["puntaje"]


# --------------------------------------------------------------------------------------------- Grid y Random Search

def grid_search(ev: Evaluador, tarea: str, modelo: str, seed: int = cfg.SEED) -> dict:
    """Recorre la rejilla explícita en orden hasta agotar el presupuesto."""
    for params in configuraciones_rejilla(tarea, modelo):
        if ev.restantes <= 0:
            break
        ev(params)
    return {}


def random_search(ev: Evaluador, tarea: str, modelo: str, seed: int = cfg.SEED) -> dict:
    """Muestreo uniforme (log-uniforme en las escalas logarítmicas) del espacio continuo."""
    espacio = ESPACIOS[(tarea, modelo)]
    rng = np.random.default_rng(seed)
    intentos = 0
    while ev.restantes > 0 and intentos < 20 * ev.presupuesto:
        ev(decodificar(espacio, rng.random(len(espacio))))
        intentos += 1
    return {}


# ------------------------------------------------------------------------------------- Bayesiana (Optuna con TPE)

def _sugerir(trial, nombre, e):
    if e["tipo"] == "cat":
        return trial.suggest_categorical(nombre, e["opciones"])
    if e["tipo"] == "int":
        return trial.suggest_int(nombre, e["bajo"], e["alto"], log=e["log"])
    return trial.suggest_float(nombre, e["bajo"], e["alto"], log=e["log"])


def bayesiana(ev: Evaluador, tarea: str, modelo: str, seed: int = cfg.SEED, n_iniciales: int = 5) -> dict:
    """Optimización bayesiana con Optuna y el estimador de Parzen en árbol (TPE).

    Modelo sustituto: TPE modela por separado la densidad de las configuraciones buenas, l(x), y la de las malas,
    g(x), separadas por el cuantil gamma del puntaje observado. Función de adquisición: Mejora Esperada (EI); bajo
    TPE, maximizar la EI equivale a maximizar l(x)/g(x) (Bergstra et al., 2011), evaluada sobre 24 candidatos
    muestreados de l(x). Las ``n_iniciales`` primeras configuraciones son aleatorias (exploración pura).
    """
    espacio = ESPACIOS[(tarea, modelo)]
    muestreador = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_iniciales, n_ei_candidates=24)
    estudio = optuna.create_study(direction="maximize", sampler=muestreador)

    def objetivo(trial):
        """Función objetivo de Optuna: evalúa la configuración sugerida; si falla devuelve -1e6 para que TPE la descarte."""
        v = ev({n: _sugerir(trial, n, e) for n, e in espacio.items()})
        return v if np.isfinite(v) else -1e6

    intentos = 0
    while ev.restantes > 0 and intentos < 5 * ev.presupuesto:
        estudio.optimize(objetivo, n_trials=1)
        intentos += 1
    return {"n_trials": len(estudio.trials)}


# ----------------------------------------------------------------------------------------------- Genético (DEAP)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMax)

POBLACION, P_CRUCE, P_MUTACION, TORNEO, ELITE = 6, 0.7, 0.3, 3, 1


def _diversidad(pob) -> float:
    G = np.asarray([list(ind) for ind in pob], dtype=float)
    if len(G) < 2:
        return 0.0
    d = [np.linalg.norm(G[i] - G[j]) for i in range(len(G)) for j in range(i + 1, len(G))]
    return float(np.mean(d) / math.sqrt(G.shape[1]))


def genetico(ev: Evaluador, tarea: str, modelo: str, seed: int = cfg.SEED) -> dict:
    """Algoritmo genético con DEAP sobre genes en [0, 1] que se decodifican al espacio de búsqueda.

    - Selección: torneo de tamaño 3 (presión selectiva moderada, robusta a la escala del puntaje).
    - Cruce: binario simulado acotado (SBX, eta = 15) con probabilidad 0,7.
    - Mutación: polinomial acotada (eta = 20, probabilidad por gen 1/d) con probabilidad 0,3 por individuo.
    - Elitismo: el mejor individuo pasa intacto a la siguiente generación.
    - Población: 6 individuos; se generan generaciones hasta agotar el presupuesto de configuraciones distintas
      (los clones ya evaluados salen del caché y no consumen presupuesto).

    Returns
    -------
    dict
        ``generaciones``: lista con la diversidad genética (distancia media entre individuos, normalizada por
        sqrt(d)), el mejor y el promedio de aptitud de cada generación.
    """
    espacio = ESPACIOS[(tarea, modelo)]
    d = len(espacio)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / d)
    tb.register("select", tools.selTournament, tournsize=TORNEO)

    def evaluar(ind):
        """Asigna como aptitud del individuo el puntaje de validación interna de la configuración que codifica."""
        ind.fitness.values = (ev(decodificar(espacio, np.asarray(ind))),)

    pob = [creator.Individuo(rng.random(d).tolist()) for _ in range(POBLACION)]
    for ind in pob:
        evaluar(ind)
    registro = []

    def anotar(gen):
        """Registra la diversidad, la mejor aptitud y la aptitud media de la generación."""
        apt = [i.fitness.values[0] for i in pob if np.isfinite(i.fitness.values[0])]
        registro.append({"generacion": gen, "n_eval": len(ev.historial), "diversidad": _diversidad(pob),
                         "mejor": max(apt) if apt else np.nan, "media": float(np.mean(apt)) if apt else np.nan})

    anotar(0)
    gen = 0
    while ev.restantes > 0 and gen < 50:
        gen += 1
        elite = [tb.clone(i) for i in tools.selBest(pob, ELITE)]
        hijos = [tb.clone(i) for i in tb.select(pob, POBLACION - ELITE)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if random.random() < P_CRUCE:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for h in hijos:
            if random.random() < P_MUTACION:
                tb.mutate(h)
                del h.fitness.values
        for h in hijos:
            if not h.fitness.valid:
                evaluar(h)
        pob = elite + hijos
        anotar(gen)
    return {"generaciones": registro}


OPTIMIZADORES = {"grid": grid_search, "random": random_search, "bayes": bayesiana, "genetico": genetico}


# ----------------------------------------------------------------------------- Multi-fidelidad: Successive Halving

def distribuciones_scipy(espacio: dict, excluir=()) -> dict:
    """Convierte un espacio al formato de ``ParameterSampler`` (distribuciones de SciPy y listas)."""
    dist = {}
    for n, e in espacio.items():
        if n in excluir:
            continue
        if e["tipo"] == "cat":
            dist[n] = e["opciones"]
        elif e["tipo"] == "int":
            dist[n] = stats.randint(e["bajo"], e["alto"] + 1) if not e["log"] else \
                [int(round(v)) for v in np.geomspace(e["bajo"], e["alto"], 25)]
        else:
            dist[n] = stats.loguniform(e["bajo"], e["alto"]) if e["log"] else stats.uniform(e["bajo"], e["alto"] - e["bajo"])
    return dist
