"""Ensambla las fuentes del Jupyter Book del cuarto entregable a partir del cuaderno ejecutado.

El cuaderno sigue la numeración de capítulos de la guía (cuadro 4), así que cada sección ``## N.`` es un capítulo del
libro: 00_introduccion (raíz), 01_correcciones, 02_datos, 03_benchmarks, 04_literatura, 05_modelo_original,
06_validacion, 07_dashboard, 08_conclusiones y 09_reproducibilidad. Además:

- sube un nivel los encabezados de cada capítulo y les agrega una etiqueta MyST (``sec-6-2`` para la sección 6.2);
- convierte en referencias cruzadas las menciones a secciones de este cuaderno («sección 6.2»), pero no las que remiten
  a la guía o a otros entregables;
- enlaza el año de cada cita «Autor (año)» o «(Autor, año)» con su entrada de ``references.bib`` (``{cite:year}``), y
  agrega al capítulo 9 la bibliografía generada por ``sphinxcontrib-bibtex``;
- oculta el código de cada celda detrás de un botón (etiqueta ``hide-input``), con sus salidas visibles.

El libro no vuelve a ejecutar el cuaderno (``execute_notebooks: off``): usa las salidas de ``construir_cuaderno.py
--ejecutar``, que es el que reproduce todas las tablas y figuras.

Uso (desde la raíz del proyecto, después de ejecutar el cuaderno):
    python scripts/construir_libro.py --libro <carpeta book del repositorio>
    jupyter-book build <carpeta book>
"""

import argparse
import copy
import re
import shutil
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
CUADERNO = RAIZ / "Proyecto Final - Cuarto Entregable - Bank Marketing.ipynb"
BIB = RAIZ / "resultados" / "literatura" / "references.bib"
CAPITULOS = {0: "00_introduccion", 1: "01_correcciones", 2: "02_datos", 3: "03_benchmarks", 4: "04_literatura",
             5: "05_modelo_original", 6: "06_validacion", 7: "07_dashboard", 8: "08_conclusiones",
             9: "09_reproducibilidad"}
REPOSITORIO = "https://github.com/DTL-DA/proyecto-final-bank-marketing"

CONFIG = f'''title: "Proyecto final · Cuarto entregable · Bank Marketing"
author: Donny Torres · Maestría en Analítica de Datos, Universidad del Norte
copyright: "2026"
only_build_toc_files: true

execute:
  execute_notebooks: "off"

parse:
  myst_enable_extensions:
    - dollarmath
    - amsmath
    - colon_fence
    - html_image

bibtex_bibfiles:
  - references.bib

repository:
  url: {REPOSITORIO}
  path_to_book: book
  branch: main

html:
  use_repository_button: true
  use_issues_button: false

sphinx:
  config:
    language: es
    nb_merge_streams: true
    nb_render_markdown_format: myst
    nb_code_prompt_show: "Mostrar el código de la celda"
    nb_code_prompt_hide: "Ocultar el código de la celda"
    bibtex_reference_style: author_year
    suppress_warnings: ["myst.header", "bibtex.duplicate_citation"]
    html_theme_options:
      show_toc_level: 2
      use_download_button: true
      use_fullscreen_button: true
      home_page_in_toc: true
'''


def subir_encabezados(texto):
    """Sube un nivel cada encabezado (## -> #), sin tocar los bloques de código."""
    salida, en_codigo = [], False
    for linea in texto.split("\n"):
        if linea.startswith("```"):
            en_codigo = not en_codigo
        if not en_codigo and re.match(r"^#{2,6} ", linea):
            linea = linea[1:]
        salida.append(linea)
    return "\n".join(salida)


ENCABEZADO = re.compile(r"^(#{1,5}) (\d+(?:\.\d+){0,2})\.? ", re.M)


def etiquetar(texto, etiquetas):
    """Antepone una etiqueta MyST a cada encabezado numerado y la registra."""
    def poner(m):
        etiqueta = "sec-" + m.group(2).replace(".", "-")
        etiquetas.add(etiqueta)
        return f"({etiqueta})=\n{m.group(0)}"
    return ENCABEZADO.sub(poner, texto)


REF = re.compile(r"([Ss]ecci(?:ón|ones)\s+)(\d+(?:\.\d+){1,2}(?:(?:,\s*|\s+(?:y|a|e|o)\s+)\d+(?:\.\d+){1,2})*)")
AJENA = re.compile(r"gu[ií]a|[Ee]ntregable|\bE[123]\b|cuaderno de|su cuaderno|libro del")


def enlazar_secciones(texto, etiquetas):
    """Convierte «sección 6.2» en una referencia cruzada si la sección es de este libro."""
    def cambiar(m):
        antes = texto[:m.start()]
        corte = max(max(antes.rfind(c) for c in "|;(\n"), antes.rfind(". "))
        if AJENA.search(antes[corte + 1:]) or re.match(r"\s*(del|de la|de su|de ese|de dicho)\b", texto[m.end():]):
            return m.group(0)

        def ref(n):
            etiqueta = "sec-" + n.group(0).replace(".", "-")
            return f"{{ref}}`{n.group(0)} <{etiqueta}>`" if etiqueta in etiquetas else n.group(0)
        return m.group(1) + re.sub(r"\d+(?:\.\d+){1,2}", ref, m.group(2))
    return REF.sub(cambiar, texto)


def claves_bib():
    """(apellido del primer autor, año) -> clave de references.bib; si se repiten, gana la primera entrada."""
    texto = BIB.read_text(encoding="utf-8")
    claves = {}
    for bloque in re.split(r"\n(?=@)", texto):
        m = re.match(r"@\w+\{([^,]+),", bloque.strip())
        autor = re.search(r"author\s*=\s*\{(.+?)\}\s*,", bloque, re.S)
        anio = re.search(r"year\s*=\s*\{?(\d{4})", bloque)
        if not (m and autor and anio):
            continue
        primero = autor.group(1).split(" and ")[0].strip()
        if "," not in primero or primero.count(",") > 1:
            continue  # formatos de autor que no son «Apellido, Nombre»
        apellido = primero.split(",")[0].strip("{} ")
        claves.setdefault((apellido, anio.group(1)), m.group(1).strip())
    return claves


def enlazar_citas(texto, claves):
    """Enlaza el año de «Autor (año)», «(Autor, año)» y «(Autor, año, ...» con su entrada de references.bib."""
    for (apellido, anio), clave in claves.items():
        a = re.escape(apellido)
        rol = f"{{cite:year}}`{clave}`"
        texto = re.sub(rf"(\b{a}\b[^()\n`]{{0,60}}?) \({anio}\)", rf"\1 ({rol})", texto)
        texto = re.sub(rf"\((\b{a}\b[^()\n`]{{0,60}}?), {anio}([),;])", rf"(\1, {rol}\2", texto)
    return texto


def fuera_de_codigo(funcion):
    """Aplica ``funcion`` solo al texto fuera de los bloques de código (```...```), que se dejan intactos."""
    def aplicar(texto):
        partes = re.split(r"(```.*?```)", texto, flags=re.S)
        return "".join(p if p.startswith("```") else funcion(p) for p in partes)
    return aplicar


def inicio_seccion(c):
    fuente = c.source
    if c.cell_type == "code" and c.get("outputs"):
        fuente = "".join(o.get("data", {}).get("text/markdown", "") for o in c.outputs if "data" in o)
    return re.search(r"^## (\d+)\. ", fuente, re.M) if c.cell_type in ("markdown", "code") else None


def transformar(c, funcion):
    if c.cell_type == "markdown":
        c.source = funcion(c.source)
    elif c.cell_type == "code":
        for o in c.get("outputs", []):
            if "data" in o and "text/markdown" in o["data"]:
                o["data"]["text/markdown"] = funcion(o["data"]["text/markdown"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libro", required=True, help="Carpeta de destino de las fuentes del libro (book/)")
    libro = Path(ap.parse_args().libro).resolve()
    if libro.exists():
        shutil.rmtree(libro)
    libro.mkdir(parents=True)
    cuaderno = nbf.read(CUADERNO, as_version=4)

    inicios = [i for i, c in enumerate(cuaderno.cells) if i > 0 and inicio_seccion(c)]
    limites = [0] + inicios + [len(cuaderno.cells)]
    capitulos = []
    for k in range(len(limites) - 1):
        celdas = [copy.deepcopy(c) for c in cuaderno.cells[limites[k]:limites[k + 1]]]
        numero = 0 if k == 0 else int(inicio_seccion(celdas[0]).group(1))
        if numero and celdas[0].cell_type == "code":
            # El encabezado lo genera el código: se pasa a una celda Markdown para que Sphinx titule el capítulo
            salida = next(o for o in celdas[0].outputs if "data" in o and "text/markdown" in o["data"])
            titulo, _, resto = salida["data"]["text/markdown"].partition("\n")
            salida["data"]["text/markdown"] = resto.lstrip("\n")
            celdas.insert(0, nbf.v4.new_markdown_cell(titulo))
        capitulos.append((numero, celdas))

    etiquetas = set()
    for numero, celdas in capitulos:
        for c in celdas:
            if numero:
                transformar(c, subir_encabezados)
            transformar(c, lambda t: etiquetar(t, etiquetas))
            if c.cell_type == "code":
                c.metadata["tags"] = sorted(set(c.metadata.get("tags", [])) | {"hide-input"})
    claves = claves_bib()
    for numero, celdas in capitulos:
        referencias = False
        for c in celdas:
            referencias = referencias or "9.3 Referencias" in (c.source if c.cell_type == "markdown" else "")
            if not referencias:  # la lista de referencias del capítulo 9 se deja tal cual
                transformar(c, fuera_de_codigo(lambda t: enlazar_citas(enlazar_secciones(t, etiquetas), claves)))
        if numero == 9:
            celdas.append(nbf.v4.new_markdown_cell(
                "## Entradas BibTeX citadas\n\nLas citas del libro enlazan a estas entradas de `references.bib`, "
                "generadas con `sphinxcontrib-bibtex`.\n\n```{bibliography}\n```"))
        capitulo = nbf.v4.new_notebook(cells=celdas)
        capitulo.metadata = copy.deepcopy(cuaderno.metadata)
        nbf.write(capitulo, libro / f"{CAPITULOS[numero]}.ipynb")
        print(f"  {CAPITULOS[numero]}.ipynb ({len(celdas)} celdas)")

    (libro / "_config.yml").write_text(CONFIG, encoding="utf-8")
    toc = "format: jb-book\nroot: 00_introduccion\nchapters:\n" + "".join(
        f"  - file: {CAPITULOS[n]}\n" for n in range(1, 10))
    (libro / "_toc.yml").write_text(toc, encoding="utf-8")
    # Los meses que devuelve doi.org vienen sin llaves (month=June) y pybtex los toma como macros no definidas
    bib = re.sub(r"(month\s*=\s*)([A-Za-z]+)(\s*[,}])", r"\1{\2}\3", BIB.read_text(encoding="utf-8"))
    (libro / "references.bib").write_text(bib, encoding="utf-8")
    if (RAIZ / "imagenes").exists():
        shutil.copytree(RAIZ / "imagenes", libro / "imagenes")
    print(f"Libro en {libro}: {len(capitulos)} capítulos, {len(etiquetas)} etiquetas de sección, "
          f"{len(claves)} referencias enlazables")


if __name__ == "__main__":
    main()
