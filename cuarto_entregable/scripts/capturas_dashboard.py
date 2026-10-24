"""Captura las tres páginas del dashboard (Edge sin ventana) para el capítulo 7 del Jupyter Book.

El dashboard debe estar corriendo (``python app.py`` en su carpeta, o la URL pública de Render). Cada captura se recorta
para quitar el fondo vacío que queda debajo del pie de página.

Uso (desde la raíz del proyecto):
    python scripts/capturas_dashboard.py --url http://127.0.0.1:8050 [--destino imagenes]
"""

import argparse
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
PAGINAS = {"dashboard_contexto": ("", 4200), "dashboard_eda": ("eda", 5200), "dashboard_modelos": ("modelos", 12000)}


def esperar(url, segundos=180):
    limite = time.time() + segundos
    while time.time() < limite:
        try:
            if urllib.request.urlopen(url, timeout=10).status == 200:
                return
        except Exception:  # noqa: BLE001 - el servidor todavía no responde
            time.sleep(3)
    raise TimeoutError(url)


def recortar(ruta):
    """Quita el fondo vacío debajo del último contenido de la página."""
    imagen = Image.open(ruta).convert("RGB")
    datos = np.asarray(imagen).astype(int)
    contraste = datos.max(axis=1) - datos.min(axis=1)
    filas = np.where(contraste.max(axis=1) > 12)[0]
    imagen.crop((0, 0, imagen.width, min(imagen.height, int(filas.max()) + 40))).save(ruta, optimize=True)


def dividir(ruta, partes=2):
    """Divide una captura larga en partes, cortando por filas de fondo (entre tarjetas) cerca de cada corte ideal."""
    imagen = Image.open(ruta).convert("RGB")
    datos = np.asarray(imagen).astype(int)
    vacias = np.where((datos.max(axis=1) - datos.min(axis=1)).max(axis=1) <= 12)[0]
    cortes = [0]
    for k in range(1, partes):
        ideal = k * imagen.height // partes
        cortes.append(int(vacias[np.argmin(np.abs(vacias - ideal))]))
    cortes.append(imagen.height)
    for k in range(partes):
        imagen.crop((0, cortes[k], imagen.width, cortes[k + 1])).save(
            ruta.with_name(f"{ruta.stem}_{k + 1}.png"), optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8050")
    ap.add_argument("--destino", default=str(RAIZ / "imagenes"))
    a = ap.parse_args()
    destino = Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    esperar(a.url)
    for nombre, (ruta, alto) in PAGINAS.items():
        archivo = destino / f"{nombre}.png"
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={archivo}",
                        f"--window-size=1400,{alto}", "--virtual-time-budget=45000", f"{a.url.rstrip('/')}/{ruta}"],
                       check=True, timeout=600, capture_output=True)
        recortar(archivo)
        if nombre == "dashboard_modelos":
            dividir(archivo)
        print(f"{archivo.name}: {Image.open(archivo).size}", flush=True)


if __name__ == "__main__":
    main()
