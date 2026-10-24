"""Arma el cuaderno del cuarto entregable a partir de las celdas de ``scripts/celdas`` y, opcionalmente, lo ejecuta.

Uso (desde la raíz del proyecto):
    python scripts/construir_cuaderno.py [--ejecutar] [--permitir-errores]
"""

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

from celdas import e4_s0, e4_s1, e4_s2, e4_s3, e4_s4, e4_s5, e4_s6, e4_s7_s9  # noqa: E402

NOMBRE = "Proyecto Final - Cuarto Entregable - Bank Marketing.ipynb"


def md(t):
    return nbformat.v4.new_markdown_cell(t)


def code(t):
    return nbformat.v4.new_code_cell(t)


def construir() -> nbformat.NotebookNode:
    s0, s1, s2, s3, s4, s5, s6, s9 = e4_s0, e4_s1, e4_s2, e4_s3, e4_s4, e4_s5, e4_s6, e4_s7_s9
    celdas = [
        md(s0.ENCABEZADO), md(s0.CONFIG_MD), code(s0.CONFIG_CODE), md(s0.CARGA_MD), code(s0.CARGA_CODE),
        code(s0.RESUMEN_CODE),
        md(s1.S11_MD), code(s1.S11_CODE), md(s1.S11B_MD), code(s1.S11B_CODE), md(s1.S12_MD), code(s1.S12_CODE),
        md(s2.S21_MD), code(s2.S21_CODE), md(s2.S22_MD), code(s2.S22_CODE),
        md(s3.S3_MD), code(s3.S3_CODE),
        md(s4.S41_MD), md(s4.S42_MD), code(s4.S42_CODE), md(s4.S43_MD), code(s4.S44_CODE),
        md(s5.S51_MD), md(s5.S52_MD), md(s5.S53_MD), md(s5.S54_MD), md(s5.S55_MD), code(s5.S55_CODE),
        md(s5.S56_MD), code(s5.S56_CODE), md(s5.S57_MD), code(s5.S57_CODE), md(s5.S58_MD), code(s5.S58_CODE),
        md(s5.S59_MD), code(s5.S59_CODE),
        md(s6.S61_MD), md(s6.S62_MD), code(s6.S62_CODE), md(s6.S63_MD), code(s6.S63_CODE), md(s6.S64_MD),
        code(s6.S64_CODE), md(s6.S65_MD), code(s6.S65_CODE), md(s6.S66_MD), code(s6.S66_CODE), md(s6.S67_MD),
        code(s6.S67_CODE), md(s6.S68_MD), code(s6.S68_CODE), md(s6.S69_MD), code(s6.S69_CODE), md(s6.S610_MD),
        code(s6.S610_CODE), md(s6.S611_MD), code(s6.S611_CODE), code(s6.S612_CODE), md(s6.S613_MD), code(s6.S613_CODE),
        md(s9.S71_MD), code(s9.S71_CODE), md(s9.S72_MD), code(s9.S72_CODE), md(s9.S73_MD), code(s9.S8_CODE),
        md(s9.S91_MD), code(s9.S92_CODE), md(s9.S93_MD),
    ]
    nb = nbformat.v4.new_notebook()
    nb.cells = celdas
    nb.metadata = {"kernelspec": {"display_name": "Python (bm_e3)", "language": "python", "name": "bm_e3"},
                   "language_info": {"name": "python"}}
    nbformat.validator.normalize(nb)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ejecutar", action="store_true")
    ap.add_argument("--permitir-errores", action="store_true")
    args = ap.parse_args()
    nb = construir()
    destino = RAIZ / NOMBRE
    nbformat.write(nb, destino)
    print(f"Cuaderno escrito: {destino.name} ({len(nb.cells)} celdas)")
    if args.ejecutar:
        cliente = NotebookClient(nb, timeout=3600, kernel_name="bm_e3", allow_errors=args.permitir_errores,
                                 resources={"metadata": {"path": str(RAIZ)}})
        try:
            cliente.execute()
        finally:
            nbformat.write(nb, destino)
        errores = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"
                   and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
        print(f"Cuaderno ejecutado · celdas con error: {errores if errores else 'ninguna'}")


if __name__ == "__main__":
    main()
