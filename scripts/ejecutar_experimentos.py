"""Ejecuta en paralelo todas las combinaciones de la validación anidada y consolida la tabla maestra.

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_experimentos.py            # todas las combinaciones (reanuda las pendientes)
    python scripts/ejecutar_experimentos.py --nucleos 8
Cada tarea queda guardada en resultados/corridas, de modo que una ejecución interrumpida continúa donde quedó.
"""

import argparse
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from joblib import Parallel, delayed  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import (consolidar, ejecutar_halving, ejecutar_tarea, guardar_consolidado, id_tarea,  # noqa: E402
                        lista_combinaciones)

COSTO = {"bosque": 6, "xgboost": 5, "knn": 4, "svm": 3, "svr": 3, "arbol": 2, "logistica": 2, "lasso": 1,
         "ridge": 1, "naive_bayes": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg.fijar_semillas()
    tareas = [(c, k) for c in lista_combinaciones() for k in range(cfg.K_EXTERNO)]
    pendientes = [(c, k) for c, k in tareas if not (cfg.RUTA_CORRIDAS / f"{id_tarea(*c, k)}.pkl").exists()]
    pendientes.sort(key=lambda t: (-COSTO[t[0][1]], -t[1]))
    halving = [(m, k) for m in ("xgboost", "bosque") for k in range(cfg.K_EXTERNO)
               if not (cfg.RUTA_CORRIDAS / f"{id_tarea('clf', m, 'ninguno', 'halving', k)}.pkl").exists()]
    print(f"Combinaciones: {len(lista_combinaciones())} · tareas: {len(tareas)} · pendientes: {len(pendientes)} "
          f"· halving pendientes: {len(halving)} · núcleos: {args.nucleos}", flush=True)
    t = time.perf_counter()
    Parallel(n_jobs=args.nucleos, backend="loky", verbose=10)(
        [delayed(ejecutar_tarea)(*c, k) for c, k in pendientes] + [delayed(ejecutar_halving)(m, k) for m, k in halving])
    print(f"Tiempo total: {(time.perf_counter() - t) / 60:.1f} min", flush=True)
    guardar_consolidado(consolidar())
    print("Tabla maestra guardada en resultados/", flush=True)


if __name__ == "__main__":
    main()
