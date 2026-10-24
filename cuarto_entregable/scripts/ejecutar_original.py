"""Validación anidada, modelo final y ablación del modelo original (L2E) en Bank Marketing.

Ejecuta, en paralelo y con registro por tarea (reanudable):

1. Las 7 variantes de ``protocolo_original.VARIANTES`` (modelo completo y ablaciones) en los 5 pliegues externos.
2. El modelo completo con las semillas 1, 2 y 3 del optimizador (estabilidad ante la semilla).
3. El modelo final de cada variante (optimización con todo el entrenamiento y evaluación única en prueba).

Uso (desde la raíz del proyecto):
    python scripts/ejecutar_original.py [--nucleos 10]
"""

import argparse
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from joblib import Parallel, delayed  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.protocolo_original import VARIANTES, ajustar_original, ejecutar_original  # noqa: E402

SEMILLAS_OPT = (1, 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    args = ap.parse_args()
    cfg.fijar_semillas()
    trabajos = [delayed(ejecutar_original)(v, k) for v in VARIANTES for k in range(cfg.K_EXTERNO)]
    trabajos += [delayed(ejecutar_original)("l2e", k, seed=s) for s in SEMILLAS_OPT for k in range(cfg.K_EXTERNO)]
    trabajos += [delayed(ajustar_original)(v) for v in VARIANTES]
    trabajos += [delayed(ajustar_original)("l2e", seed=s) for s in SEMILLAS_OPT]
    Parallel(n_jobs=args.nucleos, backend="loky", verbose=5)(trabajos)
    print("Modelo original: validación anidada, ablación, semillas y modelos finales listos", flush=True)


if __name__ == "__main__":
    main()
