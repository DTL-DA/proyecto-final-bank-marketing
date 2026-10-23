"""Fase final: consolidación, finalistas en prueba, recalibración, semillas y experimentos computacionales.

Uso (desde la raíz del proyecto, después de ejecutar_experimentos.py):
    python scripts/ejecutar_finales.py                 # finalistas, semillas y recalibración
    python scripts/ejecutar_finales.py --computacional # además, los experimentos de la sección 4 (máquina libre)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from bm import computacional as comp  # noqa: E402
from bm import config as cfg  # noqa: E402
from bm.anidada import consolidar, ejecutar_tarea, guardar_consolidado, particion  # noqa: E402
from bm.evaluacion import (RUTA_FINALES, ajustar_finalista, elegir_finalistas, modelo_referencia_e2,  # noqa: E402
                           recalibrar)
from bm.modelos import construir_pipeline  # noqa: E402

SEMILLAS_OPT = (1, 2, 3)
SEMILLAS_MODELO = (0, 1, 2, 3, 4)


def semillas_modelo(fin: dict) -> dict:
    """Reajusta un finalista con hiperparámetros fijos y cinco semillas del modelo y del sobremuestreo."""
    from bm.anidada import metricas_clf, puntajes

    p = particion()
    filas = []
    for s in SEMILLAS_MODELO:
        m = construir_pipeline("clf", fin["modelo"], fin["balanceo"], seed=s).set_params(**fin["params"])
        m.fit(p.X_tr, p.y_tr)
        sc, es_p = puntajes(m, p.X_te)
        filas.append({"modelo": fin["modelo"], "semilla": s, **metricas_clf(p.y_te, sc, m.predict(p.X_te), es_p)})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--computacional", action="store_true")
    ap.add_argument("--nucleos", type=int, default=10)
    args = ap.parse_args()
    cfg.fijar_semillas()

    tablas = consolidar()
    guardar_consolidado(tablas)
    maestra = tablas["maestra"]
    fin_clf, fin_reg = elegir_finalistas(maestra, "clf"), elegir_finalistas(maestra, "reg")
    trabajos = [delayed(ajustar_finalista)(r.tarea, r.modelo, r.balanceo, r.optimizador)
                for r in pd.concat([fin_clf, fin_reg]).itertuples()]
    trabajos.append(delayed(modelo_referencia_e2)())
    # Sensibilidad del optimizador a la semilla: XGBoost sin balanceo, optimizadores estocásticos, todos los pliegues
    trabajos += [delayed(ejecutar_tarea)("clf", "xgboost", "ninguno", o, k, seed=s)
                 for o in ("random", "bayes", "genetico") for k in range(cfg.K_EXTERNO) for s in SEMILLAS_OPT]
    Parallel(n_jobs=args.nucleos, backend="loky", verbose=5)(trabajos)

    finales = {f"{r.tarea}_{r.modelo}": ajustar_finalista(r.tarea, r.modelo, r.balanceo, r.optimizador)
               for r in pd.concat([fin_clf, fin_reg]).itertuples()}
    finales["clf_referencia_e2"] = modelo_referencia_e2()
    clf = {k: v for k, v in finales.items() if v["tarea"] == "clf"}
    recal = dict(zip(clf, Parallel(n_jobs=args.nucleos, backend="loky")(delayed(recalibrar)(v) for v in clf.values())))
    with open(RUTA_FINALES / "recalibracion.pkl", "wb") as f:
        pickle.dump(recal, f)

    candidatos = [v for k, v in clf.items() if v["modelo"] in ("bosque", "xgboost")]
    sem = Parallel(n_jobs=args.nucleos, backend="loky")(delayed(semillas_modelo)(v) for v in candidatos)
    pd.DataFrame([f for lista in sem for f in lista]).to_parquet(cfg.RUTA_RESULTADOS / "semillas_modelo.parquet")
    print("Finalistas, recalibración y semillas listos", flush=True)

    if args.computacional:
        destino = cfg.RUTA_RESULTADOS / "computacional"
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, f in (("knn_busqueda", comp.knn_busqueda), ("knn_escalamiento", comp.knn_escalamiento),
                          ("lineales", comp.lineales_solucionadores), ("lineales_escalamiento", comp.lineales_escalamiento),
                          ("naive_bayes", comp.naive_bayes_incremental), ("xgboost", comp.xgboost_metodos),
                          ("xgboost_escalamiento", comp.xgboost_escalamiento),
                          ("xgboost_parada", comp.xgboost_parada_temprana), ("svm_escalamiento", comp.svm_escalamiento),
                          ("paralelizacion", comp.paralelizacion),
                          ("comparativa", comp.comparativa_estandar_optimizado)):
            print("Computacional:", nombre, flush=True)
            f().to_parquet(destino / f"{nombre}.parquet")
        perfil = comp.perfilar_combinacion()
        perfil["tabla"].to_parquet(destino / "perfil_tabla.parquet")
        comp.agrupar_perfil(perfil["tabla"], perfil["total_s"]).to_parquet(destino / "perfil_componentes.parquet")
        with open(destino / "perfil.pkl", "wb") as f:
            pickle.dump({k: v for k, v in perfil.items() if k != "tabla"}, f)
        print("Experimentos computacionales listos", flush=True)


if __name__ == "__main__":
    main()
