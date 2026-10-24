"""Validación anidada de los 7 modelos base y del modelo original en los tres conjuntos adicionales.

Para cada conjunto (airlines, electricity, ozone) y cada pliegue externo se ejecutan los 7 finalistas de Bank
Marketing (mismo balanceo y optimizador) y tres variantes del L2E (completo, sin actualizar pendientes y estático).
Cada tarea queda en ``resultados/corridas_adicionales`` y la tabla resumen en ``resultados/adicionales.parquet``.

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_adicionales.py [--nucleos 10]
"""

import argparse
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.adicionales import DESCRIPCION, tarea_adicional  # noqa: E402
from bm.evaluacion import elegir_finalistas  # noqa: E402

VARIANTES_ADICIONALES = ("l2e", "sin_pendientes", "estatico")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    args = ap.parse_args()
    cfg.fijar_semillas()
    finalistas = elegir_finalistas(pd.read_parquet(cfg.RUTA_RESULTADOS / "e3" / "tabla_maestra.parquet"), "clf")
    trabajos = []
    for nombre in DESCRIPCION:
        for k in range(cfg.K_EXTERNO):
            trabajos += [delayed(tarea_adicional)(nombre, r.modelo, r.balanceo, r.optimizador, k, None)
                         for r in finalistas.itertuples()]
            trabajos += [delayed(tarea_adicional)(nombre, "original", "ninguno", "bayes", k, v) for v in VARIANTES_ADICIONALES]
    filas = Parallel(n_jobs=args.nucleos, backend="loky", verbose=5)(trabajos)
    pd.DataFrame(filas).to_parquet(cfg.RUTA_RESULTADOS / "adicionales.parquet", index=False)
    print("Conjuntos adicionales listos", flush=True)


if __name__ == "__main__":
    main()
