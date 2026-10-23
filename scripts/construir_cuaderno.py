"""Arma el cuaderno del tercer entregable y, opcionalmente, lo ejecuta.

Toma las secciones 1 a 3 del cuaderno del Entregable 2 (celdas 0 a 70: encabezado, base de datos, reserva de prueba,
EDA y preprocesamiento), reemplaza el encabezado y la introducción, y agrega las secciones nuevas definidas en
``scripts/celdas``.

Uso:
    python scripts/construir_cuaderno.py --origen "<ruta al cuaderno del Entregable 2>" [--ejecutar]
"""

import argparse
import copy
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

from celdas import s0_s3, s4, s5, s6, s7, s8_s10  # noqa: E402

NOMBRE = "Proyecto Final - Tercer Entregable - Bank Marketing.ipynb"


def md(t):
    return nbformat.v4.new_markdown_cell(t)


def code(t):
    return nbformat.v4.new_code_cell(t)


def construir(origen: Path) -> nbformat.NotebookNode:
    base = nbformat.read(origen, as_version=4)
    celdas = [copy.deepcopy(c) for c in base.cells[:71]]
    for c in celdas:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    celdas[0] = md(s0_s3.ENCABEZADO)
    celdas[1] = md(s0_s3.CONFIG_MD)
    texto = celdas[3].source
    i = texto.index("**Pregunta de investigación.**")
    j = texto.index("\n\n", i)
    celdas[3] = md(texto[:i] + "**Pregunta de investigación.** En el Entregable 2 la pregunta se limitó al modelo base; "
                   "las preguntas de este entregable, que amplían la comparación a 140 combinaciones, a la eficiencia de "
                   "los optimizadores y a una rama de regresión, están en la sección 4.1." + texto[j:])
    nuevas = [
        md("#### Módulos del tercer entregable\n\nA partir de aquí se cargan el paquete del proyecto (`src/bm`) y las "
           "librerías nuevas de este entregable. La celda va después del EDA heredado para que ninguna variable de "
           "las secciones 1 a 3 oculte las funciones del paquete."),
        code(s0_s3.CONFIG_PROYECTO),
        md(s0_s3.S38_MD), code(s0_s3.S381_CODE), md(s0_s3.S382_MD), code(s0_s3.S382_CODE),
        md(s4.S4_MD), code(s0_s3.CARGA_RESULTADOS), code(s4.S42_CODE), md(s4.S43_MD), code(s4.S43_CODE),
        md(s4.S44_MD), md(s4.S45_MD), code(s4.S45_CODE),
        md(s5.S51_MD), code(s5.S51_CODE), md(s5.S52_MD), md(s5.S53_MD), code(s5.S53_CODE), md(s5.S54_MD),
        code(s5.S54_CODE), code(s5.S54B_CODE), md(s5.S55_MD), code(s5.S55_CODE), md(s5.S56_MD), code(s5.S56_CODE),
        md(s5.S57_MD), code(s5.S57_CODE), md(s5.S58_MD), code(s5.S58_CODE),
        md(s6.S61_MD), code(s6.S61_CODE), md(s6.S62_MD), code(s6.S62_CODE), md(s6.S63_MD), code(s6.S63_CODE),
        md(s6.S64_MD), code(s6.S64_CODE), md(s6.S65_MD), code(s6.S65_CODE), md(s6.S66_MD), code(s6.S66_CODE),
        md(s6.S67_MD), code(s6.S67_CODE), md(s6.S68_MD), code(s6.S68_CODE), md(s6.S69_MD), code(s6.S69_CODE),
        md(s7.S71_MD), code(s7.S71_CODE), md(s7.S72_MD), code(s7.S72_CODE), md(s7.S73_MD), code(s7.S73_CODE),
        md(s7.S74_MD), code(s7.S74_CODE), md(s7.S75_MD), code(s7.S75_CODE), md(s7.S76_MD), code(s7.S76_CODE),
        code(s7.S76B_CODE), md(s7.S76C_MD), code(s7.S76C_CODE), code(s7.S76D_CODE), md(s7.S77_MD), code(s7.S77_CODE),
        md(s8_s10.S81_MD), code(s8_s10.S81_CODE), md(s8_s10.S82_MD), code(s8_s10.S82_CODE), md(s8_s10.S83_MD),
        code(s8_s10.S83_CODE), md(s8_s10.S84_MD), code(s8_s10.S84_CODE), code(s8_s10.S9_CODE), md(s8_s10.S10_MD),
    ]
    nb = nbformat.v4.new_notebook()
    nb.cells = celdas + nuevas
    nb.metadata = {"kernelspec": {"display_name": "Python (bm_e3 · Entregable 3)", "language": "python",
                                  "name": "bm_e3"},
                   "language_info": {"name": "python"}}
    nbformat.validator.normalize(nb)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True)
    ap.add_argument("--ejecutar", action="store_true")
    ap.add_argument("--permitir-errores", action="store_true")
    args = ap.parse_args()
    nb = construir(Path(args.origen))
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
        print("Cuaderno ejecutado")


if __name__ == "__main__":
    main()
