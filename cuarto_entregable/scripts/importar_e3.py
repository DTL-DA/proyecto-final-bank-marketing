"""Copia en formato compacto los resultados del tercer entregable que usa este cuaderno (benchmarks).

Lee la carpeta ``resultados`` del proyecto del tercer entregable y escribe en ``resultados/e3``:

- ``tabla_maestra.parquet`` y ``pliegues.parquet``: las 140 combinaciones y sus métricas por pliegue externo.
- ``oof_finalistas.parquet``: predicciones fuera de muestra del bucle externo de los 7 finalistas de clasificación.
- ``finales_clf.pkl``: puntajes de prueba y de entrenamiento, hiperparámetros y métricas de los finalistas y de la
  referencia del Entregable 2 (los pipelines pesados de Random Forest y KNN se omiten).
- ``recalibracion.pkl``: variantes de recalibración de la sección 7.4 del tercer entregable.

Uso (desde la raíz del proyecto):
    python scripts/importar_e3.py --origen "../Entregable_3_MachineLearningProject/resultados"
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402

from bm.evaluacion import elegir_finalistas  # noqa: E402

PESADOS = {"bosque", "knn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", default=str(RAIZ.parent / "Entregable_3_MachineLearningProject" / "resultados"))
    args = ap.parse_args()
    origen, destino = Path(args.origen), RAIZ / "resultados" / "e3"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ("tabla_maestra.parquet", "pliegues.parquet"):
        shutil.copy2(origen / nombre, destino / nombre)
    shutil.copy2(origen / "finales" / "recalibracion.pkl", destino / "recalibracion.pkl")

    maestra = pd.read_parquet(origen / "tabla_maestra.parquet")
    fin = elegir_finalistas(maestra, "clf")
    oof = pd.read_parquet(origen / "oof.parquet")
    claves = fin[["modelo", "balanceo", "optimizador"]].assign(tarea="clf")
    oof_fin = oof.merge(claves, on=["tarea", "modelo", "balanceo", "optimizador"])
    oof_fin.to_parquet(destino / "oof_finalistas.parquet", index=False)

    finales = {}
    for r in fin.itertuples():
        with open(origen / "finales" / f"clf_{r.modelo}_{r.balanceo}_{r.optimizador}_s42.pkl", "rb") as f:
            d = pickle.load(f)
        if r.modelo in PESADOS:
            d.pop("pipeline", None)
        finales[f"clf_{r.modelo}"] = d
    with open(origen / "finales" / "clf_referencia_e2.pkl", "rb") as f:
        finales["clf_referencia_e2"] = pickle.load(f)
    with open(destino / "finales_clf.pkl", "wb") as f:
        pickle.dump(finales, f)
    print(f"Importados: {len(maestra)} combinaciones, {len(oof_fin)} predicciones fuera de muestra de "
          f"{len(fin)} finalistas y {len(finales)} modelos finales en {destino}")


if __name__ == "__main__":
    main()
