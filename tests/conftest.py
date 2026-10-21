"""Configuración común: un solo hilo (como en la exportación de los artefactos) y la raíz del proyecto en el path."""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
