"""Optimización computacional (sección 4 de la guía): complejidad teórica frente a tiempos medidos."""

import cProfile
import io
import pstats
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from . import config as cfg
from .anidada import particion
from .datos import cv_interna
from .preprocesamiento import construir_preprocesador

warnings.filterwarnings("ignore")

COMPLEJIDAD = pd.DataFrame([
    ("KNN (búsqueda exhaustiva)", "O(1) (solo almacena)", "O(n · p) por consulta", "brute"),
    ("KNN (KD-Tree / Ball Tree)", "O(p · n log n)", "O(log n) en baja dimensión; tiende a O(n · p) si p es alto", "kd_tree / ball_tree"),
    ("KNN aproximado (FAISS HNSW)", "O(n log n · M)", "≈ O(log n) por consulta, con pérdida de exactitud", "IndexHNSWFlat"),
    ("Naive Bayes gaussiano", "O(n · p)", "O(p · c) por consulta", "fit / partial_fit"),
    ("Regresión logística (SAGA)", "O(n · p) por época", "O(p) por consulta", "solver saga"),
    ("Ridge (Cholesky)", "O(n · p² + p³)", "O(p)", "cholesky"),
    ("Ridge / Lasso (SAGA o SGD)", "O(n · p) por época", "O(p)", "saga / SGD"),
    ("Lasso (descenso por coordenadas)", "O(n · p) por pasada", "O(p)", "cd"),
    ("Árbol de decisión", "O(p · n log n)", "O(profundidad)", "CART"),
    ("Random Forest", "O(B · p' · n log n)", "O(B · profundidad)", "B árboles, p' variables por división"),
    ("XGBoost (exacto)", "O(B · p · n log n)", "O(B · profundidad)", "tree_method='exact'"),
    ("XGBoost (histogramas)", "O(B · p · n) tras discretizar en ≤ 256 bins", "O(B · profundidad)", "tree_method='hist'"),
    ("SVM con núcleo (SMO)", "O(n²) a O(n³)", "O(n_sv · p)", "SVC(kernel='rbf')"),
    ("SVM lineal (LinearSVC / SGD)", "O(n · p) por época", "O(p)", "liblinear / SGD"),
    ("Nyström + SVM lineal", "O(n · m · p + n · m² + m³)", "O(m · p)", "m componentes"),
], columns=["Modelo", "Entrenamiento", "Inferencia", "Implementación"])


def datos_transformados(n_max: int | None = None, seed: int = cfg.SEED):
    """Entrenamiento y prueba ya preprocesados (matrices densas), para medir solo el costo del modelo."""
    p = particion()
    pre = construir_preprocesador().fit(p.X_tr, p.y_tr)
    Xtr, Xte = pre.transform(p.X_tr).astype(np.float32), pre.transform(p.X_te).astype(np.float32)
    return Xtr, p.y_tr.to_numpy(), Xte, p.y_te.to_numpy(), p.r_tr.to_numpy(), p.r_te.to_numpy()


def _cronometrar(f, reps=3):
    tiempos = []
    for _ in range(reps):
        t = time.perf_counter()
        salida = f()
        tiempos.append(time.perf_counter() - t)
    return float(np.mean(tiempos)), float(np.std(tiempos)), salida


def exponente(ns, tiempos) -> float:
    """Pendiente de log(tiempo) contra log(n): exponente empírico de la curva de escalamiento."""
    return float(np.polyfit(np.log(ns), np.log(tiempos), 1)[0])


def _submuestra(n, total, seed=cfg.SEED):
    """Primeras n filas en orden temporal (conserva la estructura del problema)."""
    return np.arange(min(n, total))


# ----------------------------------------------------------------------------------------------------- KNN

def knn_busqueda(k: int = 40, reps: int = 3) -> pd.DataFrame:
    """Compara la búsqueda de vecinos exacta (brute, KD-Tree, Ball Tree, FAISS plano) y aproximada (FAISS HNSW, IVF).

    Mide construcción del índice, consulta de todo el conjunto de prueba, *recall*@k de los vecinos frente a la
    búsqueda exacta y el AUC-PR del clasificador KNN resultante (proporción de vecinos positivos).
    """
    import faiss

    faiss.omp_set_num_threads(1)
    Xtr, ytr, Xte, yte, *_ = datos_transformados()
    filas, exactos = [], None
    for alg in ("brute", "kd_tree", "ball_tree"):
        nn = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=1)
        t_ind, _, _ = _cronometrar(lambda: nn.fit(Xtr), reps)
        t_con, de, (_, ind) = _cronometrar(lambda: nn.kneighbors(Xte), 1 if alg != "brute" else reps)
        exactos = ind if alg == "brute" else exactos
        filas.append(("sklearn " + alg, "exacta", t_ind, t_con, ind))
    d = Xtr.shape[1]
    indices = {"FAISS IndexFlatL2": ("exacta", lambda: faiss.IndexFlatL2(d)),
               "FAISS IndexHNSWFlat (M = 32)": ("aproximada", lambda: faiss.IndexHNSWFlat(d, 32)),
               "FAISS IndexIVFFlat (100 listas, nprobe = 8)": ("aproximada", None)}
    for nombre, (tipo, crear) in indices.items():
        def construir():
            """Crea el índice FAISS (en IVF, entrena primero el cuantizador) y le agrega los datos de entrenamiento."""
            if crear is None:
                cuant = faiss.IndexFlatL2(d)
                idx = faiss.IndexIVFFlat(cuant, d, 100)
                idx.train(Xtr)
                idx.nprobe = 8
            else:
                idx = crear()
            idx.add(Xtr)
            return idx
        t_ind, _, idx = _cronometrar(construir, reps)
        t_con, _, (_, ind) = _cronometrar(lambda: idx.search(Xte, k), reps)
        filas.append((nombre, tipo, t_ind, t_con, ind))
    salida = []
    for nombre, tipo, t_ind, t_con, ind in filas:
        rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(ind, exactos)])
        prob = ytr[ind].mean(axis=1)
        salida.append({"Método": nombre, "Búsqueda": tipo, "Construcción (s)": t_ind, "Consulta prueba (s)": t_con,
                       "ms por consulta": 1000 * t_con / len(Xte), "Recall@k vs exacta": rec,
                       "AUC-PR (prueba)": average_precision_score(yte, prob)})
    return pd.DataFrame(salida)


def knn_escalamiento(ns=(2000, 4000, 8000, 16000, 32000), k: int = 40, n_consultas: int = 2000) -> pd.DataFrame:
    """Tiempo de consulta de brute y KD-Tree frente a n, para estimar el exponente empírico."""
    import faiss

    faiss.omp_set_num_threads(1)
    Xtr, _, Xte, *_ = datos_transformados()
    Q = Xte[:n_consultas]
    filas = []
    for n in ns:
        X = Xtr[_submuestra(n, len(Xtr))]
        for alg in ("brute", "kd_tree"):
            nn = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=1).fit(X)
            t, _, _ = _cronometrar(lambda: nn.kneighbors(Q), 2)
            filas.append({"Método": alg, "n": len(X), "Consulta (s)": t})
        idx = faiss.IndexHNSWFlat(X.shape[1], 32)
        idx.add(X)
        t, _, _ = _cronometrar(lambda: idx.search(Q, k), 2)
        filas.append({"Método": "FAISS HNSW", "n": len(X), "Consulta (s)": t})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------- Modelos lineales

def lineales_solucionadores(reps: int = 3) -> pd.DataFrame:
    """Solucionadores de Ridge, Lasso y logística: tiempo de ajuste y desempeño en prueba."""
    Xtr, ytr, Xte, yte, rtr, rte = datos_transformados()
    casos = [
        ("Ridge", "cholesky", lambda: Ridge(alpha=10, solver="cholesky"), "reg"),
        ("Ridge", "svd", lambda: Ridge(alpha=10, solver="svd"), "reg"),
        ("Ridge", "sag", lambda: Ridge(alpha=10, solver="sag", max_iter=1000, random_state=cfg.SEED), "reg"),
        ("Ridge", "saga", lambda: Ridge(alpha=10, solver="saga", max_iter=1000, random_state=cfg.SEED), "reg"),
        ("Lasso", "descenso por coordenadas", lambda: Lasso(alpha=1e-3, max_iter=5000), "reg"),
        ("Lasso", "SGD (L1)", lambda: SGDRegressor(penalty="l1", alpha=1e-3, random_state=cfg.SEED), "reg"),
        ("Logística L2", "lbfgs", lambda: LogisticRegression(C=0.01, max_iter=5000), "clf"),
        ("Logística L2", "saga", lambda: LogisticRegression(C=0.01, solver="saga", max_iter=1000, tol=1e-3), "clf"),
        ("Logística L1", "saga", lambda: LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000,
                                                            tol=1e-3), "clf"),
        ("Logística L1", "liblinear", lambda: LogisticRegression(C=0.01, l1_ratio=1.0, solver="liblinear"), "clf"),
    ]
    filas = []
    for modelo, solver, crear, tarea in casos:
        y = ytr if tarea == "clf" else rtr
        t, de, m = _cronometrar(lambda: crear().fit(Xtr, y), reps)
        if tarea == "clf":
            met, nombre = average_precision_score(yte, m.predict_proba(Xte)[:, 1]), "AUC-PR"
        else:
            met, nombre = root_mean_squared_error(rte, m.predict(Xte)), "RMSE"
        filas.append({"Modelo": modelo, "Solucionador": solver, "Ajuste (s)": t, "Desv. (s)": de,
                      "Métrica": nombre, "Valor": met})
    return pd.DataFrame(filas)


def lineales_escalamiento(ns=(4000, 8000, 16000, 32000)) -> pd.DataFrame:
    """Tiempo de ajuste de Ridge (Cholesky y SAGA) y de la logística (SAGA) frente a n."""
    Xtr, ytr, _, _, rtr, _ = datos_transformados()
    filas = []
    for n in ns:
        i = _submuestra(n, len(Xtr))
        for nombre, f in (("Ridge cholesky", lambda: Ridge(alpha=10, solver="cholesky").fit(Xtr[i], rtr[i])),
                          ("Ridge saga", lambda: Ridge(alpha=10, solver="saga", random_state=0).fit(Xtr[i], rtr[i])),
                          ("Logística saga", lambda: LogisticRegression(C=0.01, solver="saga", tol=1e-3,
                                                                        max_iter=1000).fit(Xtr[i], ytr[i]))):
            t, _, _ = _cronometrar(f, 3)
            filas.append({"Método": nombre, "n": len(i), "Ajuste (s)": t})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------------ Naive Bayes

def naive_bayes_incremental(tam_lote: int = 2000) -> pd.DataFrame:
    """``fit`` con todos los datos en memoria frente a ``partial_fit`` por lotes leídos del CSV (fuera de memoria).

    En el modo incremental el archivo se lee por trozos con ``pandas.read_csv(chunksize=...)``, se transforma con
    el preprocesador ya ajustado y se descarta cada trozo tras actualizar el modelo; la memoria pico se mide con
    ``tracemalloc``.
    """
    p = particion()
    pre = construir_preprocesador().fit(p.X_tr, p.y_tr)
    Xte = pre.transform(p.X_te)
    filas = []

    tracemalloc.start()
    t = time.perf_counter()
    Xtr = pre.transform(p.X_tr)
    m = GaussianNB().fit(Xtr, p.y_tr)
    t_fit = time.perf_counter() - t
    pico = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    filas.append({"Modo": "fit (todo en memoria)", "Tiempo (s)": t_fit, "Memoria pico (MB)": pico / 2**20,
                  "AUC-PR prueba": average_precision_score(p.y_te, m.predict_proba(Xte)[:, 1])})
    del Xtr

    tracemalloc.start()
    t = time.perf_counter()
    inc = GaussianNB()
    n_tr = len(p.X_tr)
    leidos = 0
    for trozo in pd.read_csv(cfg.RUTA_DATOS, sep=";", chunksize=tam_lote):
        trozo = trozo.loc[trozo.index.intersection(p.X_tr.index)]
        if trozo.empty:
            leidos += 1
            continue
        inc.partial_fit(pre.transform(trozo.drop(columns=[cfg.OBJETIVO])), trozo[cfg.OBJETIVO].eq("yes").astype(int),
                        classes=[0, 1])
    t_inc = time.perf_counter() - t
    pico = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    filas.append({"Modo": f"partial_fit (lotes de {tam_lote} filas leídos del CSV)", "Tiempo (s)": t_inc,
                  "Memoria pico (MB)": pico / 2**20,
                  "AUC-PR prueba": average_precision_score(p.y_te, inc.predict_proba(Xte)[:, 1])})
    return pd.DataFrame(filas).assign(n_entrenamiento=n_tr)


# --------------------------------------------------------------------------------------------------- XGBoost

def xgboost_metodos(n_arboles: int = 300, reps: int = 2) -> pd.DataFrame:
    """``exact`` frente a ``hist`` (y ``approx``), con un hilo, y prueba de aceleración por GPU si existe."""
    Xtr, ytr, Xte, yte, *_ = datos_transformados()
    filas = []
    for metodo in ("exact", "approx", "hist"):
        crear = lambda: XGBClassifier(tree_method=metodo, n_estimators=n_arboles, learning_rate=0.05, max_depth=6,
                                      n_jobs=1, random_state=cfg.SEED, verbosity=0)
        t, de, m = _cronometrar(lambda: crear().fit(Xtr, ytr), reps)
        ti, _, s = _cronometrar(lambda: m.predict_proba(Xte)[:, 1], reps)
        filas.append({"Configuración": f"tree_method='{metodo}' (CPU, 1 hilo)", "Ajuste (s)": t, "Inferencia (s)": ti,
                      "AUC-PR prueba": average_precision_score(yte, s), "AUC prueba": roc_auc_score(yte, s)})
    import shutil

    import xgboost

    # Con device='cuda' y sin GPU, XGBoost recurre a la CPU sin error; hay que verificar la GPU de forma explícita
    gpu = ("disponible" if xgboost.build_info().get("USE_CUDA") and shutil.which("nvidia-smi")
           else "no disponible: no hay GPU NVIDIA")
    return pd.DataFrame(filas).assign(GPU=gpu)


def xgboost_escalamiento(ns=(4000, 8000, 16000, 32000)) -> pd.DataFrame:
    """Tiempo de ajuste de ``exact`` y ``hist`` frente a n."""
    Xtr, ytr, *_ = datos_transformados()
    filas = []
    for n in ns:
        i = _submuestra(n, len(Xtr))
        for metodo in ("exact", "hist"):
            t, _, _ = _cronometrar(lambda: XGBClassifier(tree_method=metodo, n_estimators=200, max_depth=6, n_jobs=1,
                                                         verbosity=0).fit(Xtr[i], ytr[i]), 2)
            filas.append({"Método": metodo, "n": len(i), "Ajuste (s)": t})
    return pd.DataFrame(filas)


def xgboost_parada_temprana(max_arboles: int = 2000, paciencia: int = 50) -> pd.DataFrame:
    """*Early stopping* con un conjunto de validación independiente (el 20 % final del entrenamiento, con separación)."""
    p = particion()
    Xtr, ytr, Xte, yte, *_ = datos_transformados()
    n_val = int(0.2 * len(Xtr))
    Xa, ya, Xv, yv = Xtr[:-n_val - p.gap], ytr[:-n_val - p.gap], Xtr[-n_val:], ytr[-n_val:]
    filas = []
    for nombre, kw in (("Sin parada temprana (2000 árboles)", {}),
                       (f"Con parada temprana (paciencia {paciencia})", {"early_stopping_rounds": paciencia})):
        m = XGBClassifier(tree_method="hist", n_estimators=max_arboles, learning_rate=0.05, max_depth=4, n_jobs=1,
                          eval_metric="aucpr", random_state=cfg.SEED, verbosity=0, **kw)
        t = time.perf_counter()
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        t = time.perf_counter() - t
        s = m.predict_proba(Xte)[:, 1]
        filas.append({"Configuración": nombre, "Árboles usados": (m.best_iteration + 1) if kw else max_arboles,
                      "Ajuste (s)": t, "AUC-PR prueba": average_precision_score(yte, s)})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------------------- SVM

def svm_escalamiento(ns=(1000, 2000, 4000, 8000, 16000)) -> pd.DataFrame:
    """SVM con núcleo RBF exacto frente a LinearSVC, SGD (bisagra) y Nyström + LinearSVC al crecer n."""
    Xtr, ytr, Xte, yte, *_ = datos_transformados()
    metodos = {
        "SVC núcleo RBF (exacto)": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
        "LinearSVC": lambda: LinearSVC(C=0.1),
        "SGD (bisagra)": lambda: SGDClassifier(loss="hinge", alpha=1e-4, random_state=cfg.SEED),
        "Nyström (300) + LinearSVC": lambda: make_pipeline(Nystroem(gamma=0.03, n_components=300,
                                                                    random_state=cfg.SEED), LinearSVC(C=0.1)),
    }
    filas = []
    for n in ns:
        i = _submuestra(n, len(Xtr))
        for nombre, crear in metodos.items():
            if nombre.startswith("SVC") and n > 16000:
                continue
            t, _, m = _cronometrar(lambda: crear().fit(Xtr[i], ytr[i]), 2 if n <= 4000 else 1)
            ti, _, s = _cronometrar(lambda: m.decision_function(Xte), 1)
            filas.append({"Método": nombre, "n": len(i), "Ajuste (s)": t, "Inferencia (s)": ti,
                          "AUC prueba": roc_auc_score(yte, s), "AUC-PR prueba": average_precision_score(yte, s)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------- Paralelización

def paralelizacion(nucleos=(1, 2, 4, 8), reps: int = 2) -> pd.DataFrame:
    """Validación cruzada de un Random Forest con ``joblib`` (backends ``loky`` y ``threading``) y n_jobs variable.

    Se paraleliza entre pliegues (``cross_validate(n_jobs=...)``) y, por separado, entre árboles
    (``RandomForestClassifier(n_jobs=...)``), para comparar la granularidad de la paralelización.
    """
    p = particion()
    pre = construir_preprocesador()
    filas = []
    for backend in ("loky", "threading"):
        for n in nucleos:
            modelo = make_pipeline(pre, RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=1,
                                                               random_state=cfg.SEED))
            with parallel_backend(backend, n_jobs=n):
                t, _, _ = _cronometrar(lambda: cross_validate(modelo, p.X_tr, p.y_tr, cv=cv_interna(p.gap),
                                                              scoring="average_precision", n_jobs=n), reps)
            filas.append({"Nivel": "entre pliegues (cross_validate)", "Backend": backend, "n_jobs": n, "Tiempo (s)": t})
    for n in nucleos:
        modelo = make_pipeline(pre, RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=n,
                                                           random_state=cfg.SEED))
        t, _, _ = _cronometrar(lambda: cross_validate(modelo, p.X_tr, p.y_tr, cv=cv_interna(p.gap),
                                                      scoring="average_precision", n_jobs=1), reps)
        filas.append({"Nivel": "entre árboles (RandomForest n_jobs)", "Backend": "threading (interno)", "n_jobs": n,
                      "Tiempo (s)": t})
    df = pd.DataFrame(filas)
    base = df[df["n_jobs"] == 1].set_index(["Nivel", "Backend"])["Tiempo (s)"]
    df["Aceleración"] = [base.loc[(r.Nivel, r.Backend)] / r["Tiempo (s)"] for _, r in df.iterrows()]
    df["Eficiencia"] = df["Aceleración"] / df["n_jobs"]
    return df


# ------------------------------------------------------------------------------- Estándar frente a optimizado

def _medir(ajustar, predecir, reps=2):
    """Tiempo medio de ajuste e inferencia, y el modelo ajustado con sus puntajes en prueba."""
    t_fit, _, modelo = _cronometrar(ajustar, reps)
    t_inf, _, s = _cronometrar(lambda: predecir(modelo), reps)
    return t_fit, t_inf, s


def _memoria_aislada(i: int, n_svm: int, cola) -> None:
    """Memoria del ajuste del par ``i`` medida en un proceso nuevo, como incremento de su memoria residente (RSS).

    ``tracemalloc`` solo registra las reservas de Python y numpy; la RSS del proceso incluye también las de las
    librerías en C/C++ (XGBoost, FAISS, libsvm). Medir en un proceso recién creado evita que la memoria liberada por
    ajustes anteriores, que el sistema operativo no siempre devuelve, oculte el consumo del ajuste.
    """
    import gc

    import psutil
    from memory_profiler import memory_usage

    datos = datos_transformados()
    ajustar = _pares_comparativa(n_svm, datos)[i][3]
    gc.collect()
    base = psutil.Process().memory_info().rss / 2**20
    pico = memory_usage((ajustar, (), {}), interval=0.005, max_usage=True)
    cola.put(max(float(pico) - base, 0.0))


def _pares_comparativa(n_svm: int, datos) -> list:
    """Pares (estándar, optimizado) de la tabla comparativa sobre los mismos datos preprocesados."""
    import faiss

    faiss.omp_set_num_threads(1)
    Xtr, ytr, Xte, yte, rtr, rte = datos
    pares = []

    def knn_faiss():
        """Construye el índice HNSW de FAISS con los datos de entrenamiento (KNN aproximado)."""
        idx = faiss.IndexHNSWFlat(Xtr.shape[1], 32)
        idx.add(Xtr)
        return idx

    pares += [
        ("KNN (k = 40)", "Estándar: búsqueda exhaustiva", "O(n · p) por consulta",
         lambda: NearestNeighbors(n_neighbors=40, algorithm="brute").fit(Xtr),
         lambda m: ytr[m.kneighbors(Xte)[1]].mean(axis=1), "clf"),
        ("KNN (k = 40)", "Optimizado: FAISS HNSW (aproximado)", "≈ O(log n) por consulta",
         knn_faiss, lambda m: ytr[m.search(Xte, 40)[1]].mean(axis=1), "clf"),
        ("Ridge", "Estándar: Cholesky", "O(n · p² + p³)", lambda: Ridge(alpha=10, solver="cholesky").fit(Xtr, rtr),
         lambda m: m.predict(Xte), "reg"),
        ("Ridge", "Optimizado: SAGA", "O(n · p) por época",
         lambda: Ridge(alpha=10, solver="saga", random_state=0).fit(Xtr, rtr), lambda m: m.predict(Xte), "reg"),
        ("XGBoost (300 árboles)", "Estándar: tree_method='exact'", "O(B · p · n log n)",
         lambda: XGBClassifier(tree_method="exact", n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=1,
                               verbosity=0).fit(Xtr, ytr), lambda m: m.predict_proba(Xte)[:, 1], "clf"),
        ("XGBoost (300 árboles)", "Optimizado: tree_method='hist'", "O(B · p · n)",
         lambda: XGBClassifier(tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=1,
                               verbosity=0).fit(Xtr, ytr), lambda m: m.predict_proba(Xte)[:, 1], "clf"),
        (f"SVM RBF (n = {n_svm})", "Estándar: SVC núcleo exacto", "O(n²) a O(n³)",
         lambda: SVC(kernel="rbf", gamma="scale").fit(Xtr[:n_svm], ytr[:n_svm]), lambda m: m.decision_function(Xte),
         "clf"),
        (f"SVM RBF (n = {n_svm})", "Optimizado: Nyström (300) + LinearSVC", "O(n · m · p + n · m² + m³)",
         lambda: make_pipeline(Nystroem(gamma=0.03, n_components=300, random_state=cfg.SEED),
                               LinearSVC(C=0.1)).fit(Xtr[:n_svm], ytr[:n_svm]), lambda m: m.decision_function(Xte),
         "clf"),
    ]
    return pares


def comparativa_estandar_optimizado(n_svm: int = 8000) -> pd.DataFrame:
    """Tabla de la sección 6.9: complejidad, tiempos de ajuste e inferencia, memoria del ajuste y desempeño en prueba.

    Cada par usa los mismos datos preprocesados; la SVM con núcleo exacto se mide con las primeras ``n_svm`` filas
    porque con todo el entrenamiento su costo cuadrático a cúbico la vuelve impráctica (ver ``svm_escalamiento``).
    La memoria es el incremento de la memoria residente del proceso durante el ajuste, medido con
    ``memory_profiler`` en un proceso aislado por versión (ver ``_memoria_aislada``).
    """
    import multiprocessing as mp

    datos = datos_transformados()
    _, _, _, yte, _, rte = datos
    pares = _pares_comparativa(n_svm, datos)
    ctx = mp.get_context("spawn")
    filas = []
    for i, (modelo, version, comp_, ajustar, predecir, tarea) in enumerate(pares):
        t_fit, t_inf, s = _medir(ajustar, predecir)
        cola = ctx.Queue()
        proc = ctx.Process(target=_memoria_aislada, args=(i, n_svm, cola))
        proc.start()
        memoria = cola.get()
        proc.join()
        met = average_precision_score(yte, s) if tarea == "clf" else root_mean_squared_error(rte, s)
        filas.append({"Modelo": modelo, "Versión": version, "Complejidad teórica": comp_, "Ajuste (s)": t_fit,
                      "Inferencia (s)": t_inf, "Memoria del ajuste (MB)": memoria,
                      "Métrica": "AUC-PR" if tarea == "clf" else "RMSE", "Valor": met})
    df = pd.DataFrame(filas)
    base = df.groupby("Modelo")["Valor"].transform("first")
    df["Variación de la métrica"] = df["Valor"] - base
    return df


# ------------------------------------------------------------------------------------------- Perfilamiento

def perfilar_combinacion(tarea="clf", modelo="xgboost", balanceo="smote", optimizador="bayes", pliegue=4,
                         n_funciones: int = 25) -> dict:
    """Perfil de CPU (``cProfile``) y memoria (``memory_profiler``) de una combinación completa del bucle anidado."""
    from memory_profiler import memory_usage

    from .anidada import ejecutar_tarea

    perfil = cProfile.Profile()
    t = time.perf_counter()
    perfil.enable()
    ejecutar_tarea(tarea, modelo, balanceo, optimizador, pliegue, guardar=False)
    perfil.disable()
    total = time.perf_counter() - t
    salida = io.StringIO()
    pstats.Stats(perfil, stream=salida).sort_stats("cumulative").print_stats(n_funciones)
    est = pstats.Stats(perfil)
    filas = []
    for (archivo, linea, func), (cc, nc, tt, ct, _) in est.stats.items():
        filas.append({"funcion": func, "archivo": archivo, "llamadas": nc, "t_propio": tt, "t_acumulado": ct})
    tabla = pd.DataFrame(filas)
    memoria = memory_usage((ejecutar_tarea, (tarea, modelo, balanceo, optimizador, pliegue), {"guardar": False}),
                           interval=0.2, max_usage=True)
    return {"total_s": total, "tabla": tabla, "texto": salida.getvalue(), "memoria_pico_mb": float(memoria)}


def agrupar_perfil(tabla: pd.DataFrame, total: float) -> pd.DataFrame:
    """Agrupa el tiempo acumulado del perfil en componentes del pipeline (cuello de botella real)."""
    componentes = {
        "Ajuste de XGBoost (booster)": lambda r: r.funcion == "update" and "xgboost" in r.archivo,
        "Predicción de XGBoost": lambda r: r.funcion == "inplace_predict" and "xgboost" in r.archivo,
        "SMOTE (fit_resample)": lambda r: r.funcion == "fit_resample" and "imblearn" in r.archivo and "base" in r.archivo,
        "Preprocesamiento (ColumnTransformer)": lambda r: r.funcion == "fit_transform" and "_column_transformer" in r.archivo,
        "Muestreo TPE de Optuna": lambda r: r.funcion == "sample_independent" and "optuna" in r.archivo,
        "Cálculo de la métrica (scorer)": lambda r: r.funcion == "_score" and "_validation" in r.archivo,
    }
    filas = []
    for nombre, cond in componentes.items():
        sel = tabla[[cond(r) for r in tabla.itertuples()]]
        ct = sel["t_acumulado"].max() if len(sel) else 0.0
        filas.append({"Componente": nombre, "Tiempo acumulado (s)": ct, "% del total": 100 * ct / total})
    return pd.DataFrame(filas).sort_values("Tiempo acumulado (s)", ascending=False)
