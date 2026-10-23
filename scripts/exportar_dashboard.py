"""Exporta al dashboard predictivo los modelos finalistas, sus metadatos y los resultados precalculados.

Bloque B de la guía del tercer entregable: para cada uno de los 7 modelos de clasificación se toma la mejor
combinación (balanceo y optimizador) según el AUC-PR medio del bucle externo, se serializa el pipeline completo ya
reentrenado con todo el entrenamiento (``resultados/finales``) y se guarda un ``metadata.json`` con versiones,
semilla, hiperparámetros, esquema de entrada y métricas. El dashboard solo carga estos artefactos: no entrena ni
optimiza nada.

Uso (desde la raíz del proyecto):
    python scripts/exportar_dashboard.py --app "<carpeta del dashboard>"
"""

import argparse
import copy
import json
import pickle
import platform
import sys
import time
from datetime import date
from pathlib import Path

import os

# Un solo hilo, como en los demás scripts: con más hilos KNN resuelve distinto los empates de distancia
# (variables one-hot) y sus puntajes dejan de coincidir con los del cuaderno.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from bm import config as cfg  # noqa: E402
from bm import modelos as bm_modelos  # noqa: E402
from bm import preprocesamiento as bm_pre  # noqa: E402
from bm.datos import cargar_datos, preparar_particion, reconstruir_tiempo  # noqa: E402
from bm.evaluacion import RUTA_FINALES, elegir_finalistas  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

NOMBRE_MODELO = {"knn": "KNN", "naive_bayes": "Naive Bayes", "logistica": "Logística L1/L2",
                 "arbol": "Árbol de decisión", "bosque": "Random Forest", "xgboost": "XGBoost",
                 "svm": "SVM (Nyström + LinearSVC)"}
NOMBRE_BALANCEO = {"ninguno": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN", "class_weight": "class_weight"}
NOMBRE_OPTIMIZADOR = {"grid": "Grid Search", "random": "Random Search", "bayes": "Bayesiana (Optuna TPE)",
                      "genetico": "Genético (DEAP)"}
ETIQUETAS = {
    "age": ("Edad", "Cliente", "Años"), "job": ("Ocupación", "Cliente", ""), "marital": ("Estado civil", "Cliente", ""),
    "education": ("Nivel educativo", "Cliente", ""), "default": ("¿Crédito en mora?", "Cliente", ""),
    "housing": ("¿Crédito de vivienda?", "Cliente", ""), "loan": ("¿Crédito personal?", "Cliente", ""),
    "contact": ("Canal de contacto", "Contacto", ""), "month": ("Mes del contacto", "Contacto", ""),
    "day_of_week": ("Día de la semana", "Contacto", ""),
    "campaign": ("Contactos en esta campaña", "Contacto", "Incluye el último"),
    "pdays": ("Días desde el último contacto", "Historial", "999 = sin contacto previo"),
    "previous": ("Contactos en campañas anteriores", "Historial", ""),
    "poutcome": ("Resultado de la campaña anterior", "Historial", ""),
    "cons.price.idx": ("Índice de precios al consumidor", "Contexto económico", "Mensual"),
    "cons.conf.idx": ("Índice de confianza del consumidor", "Contexto económico", "Mensual"),
    "nr.employed": ("Número de empleados", "Contexto económico", "Trimestral"),
}
PASOS = {"cons.price.idx": 0.001, "cons.conf.idx": 0.1, "nr.employed": 0.1}


def rehubicar(obj, comp, vistos=None):
    """Reemplaza en el pipeline las clases y funciones del paquete ``bm`` por sus copias de ``src.componentes``."""
    vistos = set() if vistos is None else vistos
    if id(obj) in vistos:
        return
    vistos.add(id(obj))
    if type(obj) is bm_modelos.KNNBalanceado:
        obj.__class__ = comp.KNNBalanceado
    elif type(obj) is bm_modelos.XGBBalanceado:
        obj.__class__ = comp.XGBBalanceado
    if isinstance(obj, FunctionTransformer) and obj.func is bm_pre.pdays_a_nan:
        obj.func = comp.pdays_a_nan
    for atributo in ("steps", "transformers", "transformers_"):
        for item in getattr(obj, atributo, None) or []:
            if isinstance(item, tuple) and len(item) >= 2 and not isinstance(item[1], str):
                rehubicar(item[1], comp, vistos)


def puntaje_crudo(pipe, X):
    """Probabilidad de suscribir o, si el modelo no la tiene (SVM lineal), su función de decisión."""
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


def a_python(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [a_python(x) for x in v]
    return v


def curva_reducida(x, y, n=300):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return x[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="Carpeta raíz del dashboard")
    args = ap.parse_args()
    app = Path(args.app).resolve()
    sys.path.insert(0, str(app))
    from src import componentes as comp
    from src.explicaciones import shap_permutaciones

    (app / "artifacts").mkdir(exist_ok=True)
    (app / "data" / "resultados").mkdir(parents=True, exist_ok=True)
    cfg.fijar_semillas()
    rng = np.random.default_rng(cfg.SEED)

    df = cargar_datos()
    tiempo = reconstruir_tiempo(df)
    part = preparar_particion(df)
    X_tr, X_te, y_tr, y_te = part.X_tr, part.X_te, part.y_tr.to_numpy(), part.y_te.to_numpy()
    variables = cfg.VARS_NUM + cfg.VARS_CAT

    # ---------------------------------------------------------------- datos para el EDA y esquema de entrada
    banco = df.loc[np.r_[X_tr.index, X_te.index]].copy()
    banco["periodo"] = tiempo.loc[banco.index, "periodo"]
    banco["fecha"] = pd.to_datetime(banco["periodo"] + "-01")
    banco["conjunto"] = np.where(banco.index.isin(X_tr.index), "Entrenamiento", "Prueba")
    banco.reset_index(drop=True).to_parquet(app / "data" / "bank.parquet", index=False)

    todas = pd.concat([X_tr, X_te])
    esquema = []
    for v in variables:
        etiqueta, grupo, ayuda = ETIQUETAS[v]
        if v in cfg.VARS_NUM:
            serie = todas[v] if v != "pdays" else todas.loc[todas[v] != 999, v]
            entero = pd.api.types.is_integer_dtype(todas[v])
            # Valor por defecto: mediana del periodo de prueba (2009-2010), un cliente típico del régimen reciente
            fila = {"variable": v, "etiqueta": etiqueta, "grupo": grupo, "ayuda": ayuda, "tipo": "numerica",
                    "min": a_python(serie.min()), "max": a_python(serie.max()), "entero": bool(entero),
                    "paso": 1 if entero else PASOS[v], "defecto": a_python(X_te[v].median())}
            if v == "pdays":
                fila.update(especiales=[999], defecto=999)
        else:
            frec = todas[v].value_counts()
            fila = {"variable": v, "etiqueta": etiqueta, "grupo": grupo, "ayuda": ayuda, "tipo": "categorica",
                    "valores": frec.index.tolist(), "defecto": X_te[v].mode().iloc[0]}
        esquema.append(fila)

    maestra = pd.read_parquet(RAIZ / "resultados" / "tabla_maestra.parquet")
    finalistas = elegir_finalistas(maestra, "clf")
    f0 = finalistas.iloc[0]
    primero = pickle.load(open(RUTA_FINALES / f"clf_{f0['modelo']}_{f0['balanceo']}_{f0['optimizador']}_s{cfg.SEED}.pkl",
                               "rb"))["pipeline"]
    columnas = list(primero.feature_names_in_)
    no_usadas = {c: a_python(X_tr[c].median()) for c in columnas if c not in variables}
    json.dump({"variables": esquema, "columnas_entrada": columnas, "no_usadas": no_usadas},
              open(app / "data" / "esquema.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    fondo = X_tr.sample(100, random_state=cfg.SEED)[columnas]
    fondo.to_parquet(app / "data" / "fondo_shap.parquet")
    ida_vuelta = X_te.iloc[:400][columnas].copy()

    versiones = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "sklearn", "imblearn", "xgboost", "joblib", "scipy"):
        versiones[lib] = __import__(lib).__version__

    tabla, curvas, confusiones, calibracion, shap_filas, puntajes = [], [], [], [], [], {}
    for _, fila in finalistas.iterrows():
        m, b, o = fila["modelo"], fila["balanceo"], fila["optimizador"]
        fin = pickle.load(open(RUTA_FINALES / f"clf_{m}_{b}_{o}_s{cfg.SEED}.pkl", "rb"))
        pipe = copy.deepcopy(fin["pipeline"])
        rehubicar(pipe, comp)
        crudo = pickle.dumps(pipe)
        assert b"bm.modelos" not in crudo and b"bm.preprocesamiento" not in crudo, f"{m}: quedan referencias a bm"

        destino = app / "artifacts" / m
        destino.mkdir(exist_ok=True)
        joblib.dump(pipe, destino / "modelo.joblib", compress=3)
        cargado = joblib.load(destino / "modelo.joblib")
        s_te = puntaje_crudo(cargado, X_te)
        # Ida y vuelta exacta frente al modelo en memoria; frente a los puntajes guardados por el cuaderno solo puede
        # haber diferencias de redondeo (el núcleo de Nyström cambia en el último bit según el número de hilos).
        assert np.array_equal(s_te, puntaje_crudo(fin["pipeline"], X_te)), f"{m}: el artefacto no reproduce al modelo"
        assert np.allclose(s_te, fin["puntaje_te"], rtol=0, atol=1e-9), f"{m}: se aparta de los puntajes del cuaderno"
        assert np.array_equal(cargado.predict(X_te), fin["pred_te"]), f"{m}: cambian las clases predichas"
        relleno = X_te.copy()
        for c, v in no_usadas.items():
            relleno[c] = v
        assert np.array_equal(puntaje_crudo(cargado, relleno), s_te), f"{m}: depende de columnas no usadas"
        ida_vuelta[f"esperado__{m}"] = s_te[:400]

        platt = None
        if not fin["es_proba"]:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(fin["puntaje_tr"].reshape(-1, 1), y_tr)
            platt = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]),
                     "nota": "Escalado de Platt ajustado con los puntajes de entrenamiento del modelo final"}
        prob_te = s_te if platt is None else 1 / (1 + np.exp(-(platt["a"] * s_te + platt["b"])))

        tiempos = []
        for _ in range(3):
            t0 = time.perf_counter()
            puntaje_crudo(cargado, X_te)
            tiempos.append(time.perf_counter() - t0)
        ms_mil = 1000 * np.median(tiempos) / len(X_te) * 1000

        bins = np.minimum((prob_te * 10).astype(int), 9)
        ece = 0.0
        for k in range(10):
            sel = bins == k
            if sel.any():
                calibracion.append({"modelo": m, "bin": k, "prob_media": float(prob_te[sel].mean()),
                                    "frec_obs": float(y_te[sel].mean()), "n": int(sel.sum())})
                ece += sel.mean() * abs(prob_te[sel].mean() - y_te[sel].mean())
        fpr, tpr, _ = roc_curve(y_te, s_te)
        prec, rec, _ = precision_recall_curve(y_te, s_te)
        for tipo, (cx, cy) in (("roc", curva_reducida(fpr, tpr)), ("pr", curva_reducida(rec[::-1], prec[::-1]))):
            curvas += [{"modelo": m, "tipo": tipo, "x": float(a), "y": float(c)} for a, c in zip(cx, cy)]
        tn, fp, fn, tp = confusion_matrix(y_te, fin["pred_te"].astype(int)).ravel()
        confusiones.append({"modelo": m, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        puntajes[m] = s_te

        ext = maestra[(maestra["tarea"] == "clf") & (maestra["modelo"] == m)]
        tabla.append({
            "modelo": m, "nombre": NOMBRE_MODELO[m], "balanceo": NOMBRE_BALANCEO[b], "optimizador": NOMBRE_OPTIMIZADOR[o],
            "combinaciones": len(ext), "auc_pr_media": fila["auc_pr_media"], "auc_pr_de": fila["auc_pr_de"],
            "auc_media": fila["auc_media"], "auc_de": fila["auc_de"], "recall_media": fila["recall_media"],
            "f1_media": fila["f1_media"], "auc_pr_prueba": fin["auc_pr"], "auc_prueba": fin["auc"],
            "brier_prueba": float(brier_score_loss(y_te, prob_te)), "ece_prueba": float(ece), "inferencia_ms_mil": ms_mil})

        metadata = {
            "modelo": m, "nombre": NOMBRE_MODELO[m], "balanceo": b, "optimizador": o, "semilla": cfg.SEED,
            "hiperparametros": {k: a_python(v) for k, v in fin["params"].items()},
            "salida": "predict_proba" if fin["es_proba"] else "decision_function", "platt": platt,
            "umbral": "0,5 sobre la probabilidad" if fin["es_proba"] else "0 sobre la función de decisión",
            "versiones": versiones, "columnas_entrada": columnas, "variables_usadas": variables,
            "esquema": "data/esquema.json",
            "metricas_bucle_externo": {k: {"media": a_python(fila[f"{k}_media"]), "de": a_python(fila[f"{k}_de"])}
                                       for k in ("auc_pr", "auc", "f1", "recall", "precision", "accuracy", "brier")},
            "metricas_prueba": {k: a_python(fin[k]) for k in ("auc_pr", "auc", "f1", "recall", "precision", "accuracy")},
            "motivo_seleccion": f"Mayor AUC-PR medio del bucle externo entre las {len(ext)} combinaciones de {NOMBRE_MODELO[m]}",
            "entrenamiento": "Reentrenado con todo el conjunto de entrenamiento (2008-05 a 2009-05) con los "
                             "hiperparámetros del mismo optimizador; prueba (2009-05 a 2010-11) usada una sola vez.",
        }
        json.dump(metadata, open(destino / "metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        f = (lambda D, p=cargado, pl=platt: puntaje_crudo(p, D) if pl is None
             else 1 / (1 + np.exp(-(pl["a"] * puntaje_crudo(p, D) + pl["b"]))))
        n_filas = 60 if m == "knn" else 120
        filas_shap = X_te.iloc[rng.choice(len(X_te), n_filas, replace=False)][columnas]
        for i in range(n_filas):
            phi, _, _ = shap_permutaciones(f, filas_shap.iloc[[i]], fondo, variables, n_perm=32, semilla=i)
            shap_filas += [{"modelo": m, "fila": i, "variable": v, "phi": float(p_), "valor": str(filas_shap.iloc[i][v])}
                           for v, p_ in zip(variables, phi)]
        print(f"{m}: exportado ({(destino / 'modelo.joblib').stat().st_size / 2**20:.1f} MB), "
              f"AUC-PR prueba {fin['auc_pr']:.3f}, inferencia {ms_mil:.1f} ms por 1 000", flush=True)

    r = app / "data" / "resultados"
    pd.DataFrame(tabla).to_parquet(r / "leaderboard.parquet", index=False)
    pd.DataFrame(curvas).to_parquet(r / "curvas.parquet", index=False)
    pd.DataFrame(confusiones).to_parquet(r / "confusion.parquet", index=False)
    pd.DataFrame(calibracion).to_parquet(r / "calibracion.parquet", index=False)
    pd.DataFrame(shap_filas).to_parquet(r / "shap_global.parquet", index=False)
    pd.DataFrame(puntajes).to_parquet(r / "puntajes_prueba.parquet", index=False)
    ida_vuelta.to_parquet(app / "data" / "roundtrip.parquet")
    datos_dash = RAIZ / "dashboard" / "datos"
    pd.read_parquet(datos_dash / "rangos_cd.parquet").to_parquet(r / "cd.parquet", index=False)
    pd.read_parquet(datos_dash / "friedman_omnibus.parquet").to_parquet(r / "friedman.parquet", index=False)
    pd.read_parquet(datos_dash / "delong.parquet").to_parquet(r / "delong.parquet", index=False)

    eur = banco[banco["conjunto"] == "Entrenamiento"].groupby("periodo")["euribor3m"].median()  # como la sección 3.6.1
    contexto = {
        "registros": int(len(df)), "registros_sin_duplicados": int(len(banco)), "entrenamiento": int(len(X_tr)),
        "prueba": int(len(X_te)), "tasa_total": float((df["y"] == "yes").mean()), "tasa_entrenamiento": float(y_tr.mean()),
        "tasa_prueba": float(y_te.mean()), "periodo_inicio": banco["periodo"].min(), "periodo_fin": banco["periodo"].max(),
        "periodos": int(banco["periodo"].nunique()), "euribor_inicio": float(eur.iloc[0]), "euribor_2009_05": float(eur["2009-05"]),
        "combinaciones": int(len(maestra)), "evaluaciones": int(maestra["n_eval_total"].sum()),
        "horas_cpu": float(maestra["t_opt_total"].sum() / 3600),
        "mejor_global": f"{NOMBRE_MODELO[finalistas.iloc[0]['modelo']]} · {NOMBRE_BALANCEO[finalistas.iloc[0]['balanceo']]}"
                        f" · {NOMBRE_OPTIMIZADOR[finalistas.iloc[0]['optimizador']]}",
        "actualizado": date.today().isoformat(), "version": "1.0",
    }
    json.dump(contexto, open(app / "data" / "contexto.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Exportación completa en", app, flush=True)


if __name__ == "__main__":
    main()
