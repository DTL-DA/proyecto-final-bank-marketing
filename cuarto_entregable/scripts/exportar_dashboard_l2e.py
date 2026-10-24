"""Integra el modelo original (L2E) al dashboard: artefacto, metadatos y resultados precalculados.

Es el equivalente, para el modelo original, de ``scripts/exportar_dashboard.py`` del tercer entregable (bloque B de la
guía): toma el modelo final de ``resultados/finales_original/l2e.pkl`` (ajustado con todo el entrenamiento), le
reemplaza las referencias al paquete de investigación ``bm`` por sus copias de producción del dashboard
(``src.original_model`` y ``src.componentes``), lo serializa con ``joblib`` y verifica la prueba de ida y vuelta.
Después agrega el L2E a las tablas precalculadas del dashboard (tabla de líderes, curvas, matriz de confusión,
calibración, SHAP global, puntajes de prueba y diagrama de diferencia crítica) y exporta lo que necesita la subsección
del modelo original. El dashboard no entrena ni optimiza nada: solo carga lo que deja este script.

Se guardan dos artefactos: ``modelo.joblib``, con el estado al cierre del entrenamiento, que es el que se evaluó en la
prueba, y ``modelo_vigente.joblib``, que además asimiló las etiquetas de todos los días de la prueba y es el que el
dashboard usa para predecir el día siguiente al último cerrado.

Uso (desde la raíz del proyecto, después de ejecutar el cuaderno):
    python scripts/exportar_dashboard_l2e.py --app "<carpeta del dashboard>"
"""

import argparse
import copy
import json
import os
import pickle
import platform
import re
import sys
import time
from datetime import date
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from bm import config as cfg  # noqa: E402
from bm import estadistica as st  # noqa: E402
from bm import estadistica_e4 as st4  # noqa: E402
from bm import modelo_original as bm_original  # noqa: E402
from bm import preprocesamiento as bm_pre  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.evaluacion import elegir_finalistas  # noqa: E402
from bm.protocolo_original import con_dia, puntaje_secuencial  # noqa: E402
from celdas import e4_s4, e4_s5  # noqa: E402

RES = cfg.RUTA_RESULTADOS
NOMBRE_L2E = "L2E (modelo original)"
CORTO = {"knn": "KNN", "naive_bayes": "Naive Bayes", "logistica": "Logística L1/L2", "arbol": "Árbol de decisión",
         "bosque": "Random Forest", "xgboost": "XGBoost", "svm": "SVM", "referencia_e2": "Referencia E2",
         "l2e": NOMBRE_L2E}


def leer(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


def a_python(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [a_python(x) for x in v]
    return v


def rehubicar(obj, om, comp, vistos=None):
    """Reemplaza en el pipeline la clase del L2E y la función de ``pdays`` por sus copias del dashboard."""
    vistos = set() if vistos is None else vistos
    if id(obj) in vistos:
        return
    vistos.add(id(obj))
    if type(obj) is bm_original.LogisticaDosEscalas:
        obj.__class__ = om.LogisticaDosEscalas
    if isinstance(obj, FunctionTransformer) and obj.func is bm_pre.pdays_a_nan:
        obj.func = comp.pdays_a_nan
    for atributo in ("steps", "transformers", "transformers_"):
        for item in getattr(obj, atributo, None) or []:
            if isinstance(item, tuple) and len(item) >= 2 and not isinstance(item[1], str):
                rehubicar(item[1], om, comp, vistos)


def curva_reducida(x, y, n=300):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return x[idx], y[idx]


def tabla_markdown(texto):
    """Primera tabla Markdown de un texto como DataFrame (encabezado, separador y filas)."""
    filas = [l for l in texto.splitlines() if l.startswith("|")]
    celdas = [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", f)[1:-1]] for f in filas]
    return pd.DataFrame(celdas[2:], columns=celdas[0])


def reemplazar(r, nombre, nuevo, clave="modelo"):
    """Quita las filas del L2E de una tabla precalculada (si las hay) y agrega las nuevas."""
    ruta = r / f"{nombre}.parquet"
    viejo = pd.read_parquet(ruta)
    viejo = viejo[viejo[clave] != "l2e"] if clave in viejo.columns else viejo
    return pd.concat([viejo, nuevo], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="Carpeta raíz del dashboard")
    app = Path(ap.parse_args().app).resolve()
    sys.path.insert(0, str(app))
    from src import componentes as comp
    from src import original_model as om
    from src.explicaciones import shap_permutaciones

    r = app / "data" / "resultados"
    destino = app / "artifacts" / "l2e"
    destino.mkdir(parents=True, exist_ok=True)
    esquema = json.loads((app / "data" / "esquema.json").read_text(encoding="utf-8"))
    columnas = esquema["columnas_entrada"]
    variables = [v["variable"] for v in esquema["variables"]]
    fondo = pd.read_parquet(app / "data" / "fondo_shap.parquet")

    p = particion()
    X_tr, X_te, y_tr, y_te = p.X_tr, p.X_te, p.y_tr.to_numpy(), p.y_te.to_numpy()
    X_tr_d, X_te_d = con_dia(X_tr, p.tiempo_tr), con_dia(X_te, p.tiempo_te)
    fin = leer(RES / "finales_original" / "l2e.pkl")
    corr = [leer(RES / "corridas_original" / f"l2e_{k}.pkl") for k in range(cfg.K_EXTERNO)]
    s_te = np.asarray(fin["puntaje_te"])

    # ------------------------------------------------------------------------ artefactos y prueba de ida y vuelta
    pipe = copy.deepcopy(fin["pipeline"])
    vigente = copy.deepcopy(pipe)
    vigente[-1].actualizar(vigente[:-1].transform(X_te_d), y_te)
    dia_pred = int(vigente[-1].dia_final_) + 1
    memoria = {"modelo.joblib": copy.deepcopy(pipe), "modelo_vigente.joblib": copy.deepcopy(vigente)}
    for nombre, obj in (("modelo.joblib", pipe), ("modelo_vigente.joblib", vigente)):
        rehubicar(obj, om, comp)
        crudo = pickle.dumps(obj)
        assert b"bm.modelo_original" not in crudo and b"bm.preprocesamiento" not in crudo, f"{nombre}: quedan referencias a bm"
        joblib.dump(obj, destino / nombre, compress=3)

    cerrado = joblib.load(destino / "modelo.joblib")
    assert np.array_equal(cerrado.predict_proba(X_te_d), memoria["modelo.joblib"].predict_proba(X_te_d))
    s_disco = puntaje_secuencial(cerrado, X_te_d, y_te)
    assert np.array_equal(s_disco, puntaje_secuencial(memoria["modelo.joblib"], X_te_d, y_te)), "cambia la precuencial"
    assert np.allclose(s_disco, s_te, rtol=0, atol=1e-10), "se aparta de los puntajes del cuaderno"
    leido = joblib.load(destino / "modelo_vigente.joblib")
    ida_vuelta = pd.read_parquet(app / "data" / "roundtrip.parquet")
    X_rt = ida_vuelta[columnas].assign(dia=dia_pred)
    esperado = leido.predict_proba(X_rt)[:, 1]
    assert np.array_equal(esperado, memoria["modelo_vigente.joblib"].predict_proba(X_rt)[:, 1]), "el vigente cambia"
    relleno = X_rt.copy()
    for c_, v_ in esquema["no_usadas"].items():
        relleno[c_] = v_
    assert np.array_equal(leido.predict_proba(relleno)[:, 1], esperado), "el L2E depende de columnas no usadas"
    ida_vuelta["esperado__l2e"] = esperado
    ida_vuelta.to_parquet(app / "data" / "roundtrip.parquet")

    # ------------------------------------------------------------------------------------- metadatos del modelo
    def media_de(clave):
        v = np.array([c_[clave] for c_ in corr], dtype=float)
        return {"media": float(v.mean()), "de": float(v.std(ddof=1))}

    versiones = {"python": platform.python_version()}
    for lib in ("numpy", "pandas", "sklearn", "imblearn", "xgboost", "joblib", "scipy"):
        versiones[lib] = __import__(lib).__version__
    params = {k.split("__")[1]: float(v) for k, v in fin["params"].items()}
    metadata = {
        "modelo": "l2e", "nombre": NOMBRE_L2E, "clase": "src.original_model.LogisticaDosEscalas", "balanceo": "ninguno",
        "optimizador": "bayes", "semilla": cfg.SEED, "hiperparametros": params,
        "parametros_fijos": {"retardo": 1, "nivel_latente": True, "actualizar_nivel": True, "actualizar_pendientes": True,
                             "integrar_incertidumbre": True},
        "salida": "predict_proba", "platt": None, "umbral": "0,5 sobre la probabilidad",
        "archivo": "modelo_vigente.joblib", "dia_prediccion": dia_pred,
        "estado": {"modelo.joblib": {"ultimo_dia": int(pipe[-1].dia_final_), "periodo": str(p.tiempo_tr["periodo"].iloc[-1]),
                                     "uso": "evaluado una sola vez en la prueba, de forma precuencial"},
                   "modelo_vigente.joblib": {"ultimo_dia": int(vigente[-1].dia_final_),
                                             "periodo": str(p.tiempo_te["periodo"].iloc[-1]),
                                             "uso": "predicción del día siguiente al último cerrado"}},
        "versiones": versiones, "columnas_entrada": columnas, "variables_usadas": variables,
        "esquema": "data/esquema.json",
        "metricas_bucle_externo": {k: media_de(k) for k in ("auc_pr", "auc", "f1", "recall", "precision", "accuracy",
                                                              "brier")},
        "metricas_prueba": {k: a_python(fin[k]) for k in ("auc_pr", "auc", "f1", "recall", "precision", "accuracy",
                                                          "brier")},
        "motivo_seleccion": "Modelo original del proyecto, validado con los mismos pliegues externos, la misma semilla y "
                            "el mismo presupuesto de Optuna (20 configuraciones) que los 140 benchmarks",
        "entrenamiento": "Ajustado con todo el entrenamiento (2008-05 a 2009-05); la prueba (2009-05 a 2010-11) se "
                         "evaluó una sola vez, de forma precuencial y con un día de demora en las etiquetas.",
    }
    (destino / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- tablas comunes con los demás modelos
    t0 = []
    for _ in range(3):
        inicio = time.perf_counter()
        leido.predict_proba(X_te.assign(dia=dia_pred))
        t0.append(time.perf_counter() - inicio)
    bins = np.minimum((s_te * 10).astype(int), 9)
    calib, ece = [], 0.0
    for k in range(10):
        sel = bins == k
        if sel.any():
            calib.append({"modelo": "l2e", "bin": k, "prob_media": float(s_te[sel].mean()),
                          "frec_obs": float(y_te[sel].mean()), "n": int(sel.sum())})
            ece += sel.mean() * abs(s_te[sel].mean() - y_te[sel].mean())
    ext = metadata["metricas_bucle_externo"]
    fila = {"modelo": "l2e", "nombre": NOMBRE_L2E, "balanceo": "Sin balanceo", "optimizador": "Bayesiana (Optuna TPE)",
            "combinaciones": 1, "auc_pr_media": ext["auc_pr"]["media"], "auc_pr_de": ext["auc_pr"]["de"],
            "auc_media": ext["auc"]["media"], "auc_de": ext["auc"]["de"], "recall_media": ext["recall"]["media"],
            "f1_media": ext["f1"]["media"], "auc_pr_prueba": float(fin["auc_pr"]), "auc_prueba": float(fin["auc"]),
            "brier_prueba": float(brier_score_loss(y_te, s_te)), "ece_prueba": float(ece),
            "inferencia_ms_mil": 1000 * np.median(t0) / len(X_te) * 1000}
    lb = reemplazar(r, "leaderboard", pd.DataFrame([fila])).sort_values("auc_pr_media", ascending=False)
    lb.to_parquet(r / "leaderboard.parquet", index=False)

    curvas = []
    fpr, tpr, _ = roc_curve(y_te, s_te)
    prec, rec, _ = precision_recall_curve(y_te, s_te)
    for tipo, (cx, cy) in (("roc", curva_reducida(fpr, tpr)), ("pr", curva_reducida(rec[::-1], prec[::-1]))):
        curvas += [{"modelo": "l2e", "tipo": tipo, "x": float(a), "y": float(b)} for a, b in zip(cx, cy)]
    reemplazar(r, "curvas", pd.DataFrame(curvas)).to_parquet(r / "curvas.parquet", index=False)
    tn, fp, fn, tp = confusion_matrix(y_te, (s_te >= 0.5).astype(int)).ravel()
    reemplazar(r, "confusion", pd.DataFrame([{"modelo": "l2e", "tn": int(tn), "fp": int(fp), "fn": int(fn),
                                              "tp": int(tp)}])).to_parquet(r / "confusion.parquet", index=False)
    reemplazar(r, "calibracion", pd.DataFrame(calib)).to_parquet(r / "calibracion.parquet", index=False)
    puntajes = pd.read_parquet(r / "puntajes_prueba.parquet")
    puntajes["l2e"] = s_te
    puntajes.to_parquet(r / "puntajes_prueba.parquet", index=False)

    # SHAP global: mismo estimador y misma muestra que los demás modelos, para un cliente llamado el día siguiente
    def f(D):
        return leido.predict_proba(D.assign(dia=dia_pred))[:, 1]

    rng = np.random.default_rng(cfg.SEED)
    filas_shap = X_te.iloc[rng.choice(len(X_te), 120, replace=False)][columnas]
    shap_filas = []
    for i in range(len(filas_shap)):
        phi, _, _ = shap_permutaciones(f, filas_shap.iloc[[i]], fondo, variables, n_perm=32, semilla=i)
        shap_filas += [{"modelo": "l2e", "fila": i, "variable": v, "phi": float(p_), "valor": str(filas_shap.iloc[i][v])}
                       for v, p_ in zip(variables, phi)]
    reemplazar(r, "shap_global", pd.DataFrame(shap_filas)).to_parquet(r / "shap_global.parquet", index=False)

    # ------------------------------------------ diagrama de diferencia crítica con el modelo original incluido
    maestra = pd.read_parquet(RES / "e3" / "tabla_maestra.parquet")
    pliegues_e3 = pd.read_parquet(RES / "e3" / "pliegues.parquet")
    fin_e3 = leer(RES / "e3" / "finales_clf.pkl")
    finalistas = elegir_finalistas(maestra, "clf")
    M_pl = pd.DataFrame({"l2e": [c_["auc_pr"] for c_ in corr]})
    for fr_ in finalistas.itertuples():
        sel = pliegues_e3[(pliegues_e3["tarea"] == "clf") & (pliegues_e3["modelo"] == fr_.modelo)
                          & (pliegues_e3["balanceo"] == fr_.balanceo) & (pliegues_e3["optimizador"] == fr_.optimizador)
                          & (pliegues_e3["seed"] == cfg.SEED)].sort_values("pliegue")
        M_pl[fr_.modelo] = sel["auc_pr"].to_numpy()
    meses = p.tiempo_te["periodo"].to_numpy()
    puntajes_te = {"l2e": s_te, **{m: fin_e3[f"clf_{m}"]["puntaje_te"] for m in finalistas["modelo"]},
                   "referencia_e2": fin_e3["clf_referencia_e2"]["puntaje_te"]}
    M_mes = pd.DataFrame({k: st4.metrica_por_bloque(y_te, s_, meses) for k, s_ in puntajes_te.items()}).dropna()
    filas_cd, friedman = [], {}
    for bloques, M in (("pliegues", M_pl), ("meses_prueba", M_mes)):
        fr = st.friedman(M)
        cd = st.diferencia_critica(M.shape[1], M.shape[0])
        friedman[bloques] = {"chi2": float(fr["chi2"]), "p": float(fr["p"]), "k": int(M.shape[1]), "n": int(M.shape[0]),
                             "cd": cd, "rango_l2e": float(fr["rangos"]["l2e"]),
                             "puesto_l2e": int(list(fr["rangos"].index).index("l2e")) + 1,
                             "mejor": CORTO[fr["rangos"].index[0]]}
        filas_cd += [{"modelo": CORTO[m], "rango": float(v), "bloques": bloques, "cd": cd}
                     for m, v in fr["rangos"].items()]
    pd.DataFrame(filas_cd).to_parquet(r / "cd.parquet", index=False)

    # ------------------------------------------------------------------- subsección del modelo original
    traza = fin["traza"]
    nombres_x = pipe[:-1].get_feature_names_out()[:-1]
    xbar = cerrado[:-1].transform(X_tr_d)[:, :-1].mean(axis=0)
    obs_te = pd.DataFrame({"dia": p.tiempo_te["dia"].to_numpy(), "y": y_te,
                           "periodo": meses}).groupby("dia").agg(tasa=("y", "mean"), n=("y", "size"),
                                                                 periodo=("periodo", "first"))
    nivel_te = pd.DataFrame({"dia": traza["dia"], "prob": expit(traza["nivel"] + traza["pendientes"] @ xbar),
                             "conjunto": "Prueba (filtrado, un día de demora)"}).set_index("dia").join(obs_te)
    obs_tr = pd.DataFrame({"dia": p.tiempo_tr["dia"].to_numpy(), "y": y_tr,
                           "periodo": p.tiempo_tr["periodo"].to_numpy()}).groupby("dia").agg(
        tasa=("y", "mean"), n=("y", "size"), periodo=("periodo", "first"))
    m_ = cerrado[-1]
    nivel_tr = pd.DataFrame({"dia": m_.dias_, "prob": expit(m_.nivel_ + m_.coef_ @ xbar),
                             "conjunto": "Entrenamiento (suavizado)"}).set_index("dia").join(obs_tr)
    pd.concat([nivel_tr, nivel_te]).reset_index().to_parquet(r / "l2e_nivel.parquet", index=False)
    pd.DataFrame([{k: h[k] for k in ("n", "puntaje", "mejor", "t_acum")} for h in fin["historial"]]).to_parquet(
        r / "l2e_convergencia.parquet", index=False)
    pd.read_parquet(RES / "analisis" / "tamano.parquet").to_parquet(r / "l2e_aprendizaje.parquet", index=False)
    tabla_markdown(e4_s4.S43_MD).to_parquet(r / "l2e_literatura.parquet", index=False)
    pr = pd.read_parquet(RES / "guion" / "pruebas_r3.parquet")
    pr["Efecto"] = pr["Efecto"].str.replace("d̄", "Δ medio", regex=False)
    pr.to_parquet(r / "l2e_pruebas.parquet", index=False)
    pd.read_parquet(RES / "guion" / "conteo_r3.parquet").to_parquet(r / "l2e_conteo.parquet", index=False)
    pd.read_parquet(RES / "guion" / "ablacion.parquet").to_parquet(r / "l2e_ablacion.parquet", index=False)

    g = RES / "guion"
    r3 = json.loads((RES / "mejoras" / "resumen_r3.json").read_text(encoding="utf-8"))
    ap_sp = np.array([leer(RES / "corridas_original" / f"sin_pendientes_{k}.pkl")["auc_pr"] for k in range(cfg.K_EXTERNO)])
    ap_l2 = M_pl["l2e"].to_numpy()
    nb_sp = st4.t_nadeau_bengio(ap_sp, ap_l2, [c_["n_ent"] for c_ in corr], [c_["n_val"] for c_ in corr])
    bay_sp = st4.rangos_signo_bayes(ap_sp, ap_l2, rope=0.01)
    resumen = {
        "r3": r3, "friedman": friedman, "hiperparametros": params, "dia_prediccion": dia_pred,
        "periodo_vigente": str(p.tiempo_te["periodo"].iloc[-1]), "n_coeficientes": int(len(nombres_x)),
        "prob_media_prueba": float(s_te.mean()), "tasa_prueba": float(y_te.mean()),
        "auc_pr_prueba": float(fin["auc_pr"]), "auc_prueba": float(fin["auc"]), "brier_prueba": fila["brier_prueba"],
        "referencia_e2": {"auc_pr_prueba": float(fin_e3["clf_referencia_e2"]["auc_pr"]),
                          "auc_prueba": float(fin_e3["clf_referencia_e2"]["auc"])},
        "literatura": json.loads((g / "literatura.json").read_text(encoding="utf-8")),
        "pruebas_unitarias": json.loads((g / "pruebas_unitarias.json").read_text(encoding="utf-8")),
        "robustez": json.loads((g / "robustez.json").read_text(encoding="utf-8")),
        "ablacion_sp": {"dif": float(ap_sp.mean() - ap_l2.mean()), "gana": int((ap_sp > ap_l2).sum()),
                        "p_nb": nb_sp["p"], "p_rope": bay_sp["p_rope"], "p_mejor": bay_sp["p_a_mejor"]},
        "pseudocodigo": e4_s5.S53_MD.split("```text\n", 1)[1].split("```", 1)[0].rstrip(),
        "mejoras": "Ninguna de las cinco mejoras evaluadas (splines, Bayes empírico, recencia, ensamble y semillas) supera "
                   "la regla declarada de 0,004 en el AUC-PR externo, así que el L2E se conserva sin cambios.",
    }
    (r / "l2e_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")

    contexto = json.loads((app / "data" / "contexto.json").read_text(encoding="utf-8"))
    contexto.update(version="2.0", actualizado=date.today().isoformat(), entrega="Cuarto entregable",
                    modelo_original="L2E (logística de estado latente en dos escalas)")
    (app / "data" / "contexto.json").write_text(json.dumps(contexto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"L2E exportado: artefactos de {sum(f.stat().st_size for f in destino.glob('*.joblib')) / 1024:.0f} KB, "
          f"día de predicción {dia_pred}, AUC-PR externo {ext['auc_pr']['media']:.3f}, prueba {fin['auc_pr']:.3f}, "
          f"inferencia {fila['inferencia_ms_mil']:.1f} ms por 1 000 clientes", flush=True)


if __name__ == "__main__":
    main()
