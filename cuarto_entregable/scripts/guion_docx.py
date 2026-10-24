"""Utilidades para agregar secciones al guion de sustentación (Word) con el formato de sus cuadros.

Replica el estilo del documento: párrafos justificados (interlineado 1,25 y 7 pt después), cuadros de tres reglas al
estilo LaTeX (gruesa arriba, delgada bajo el encabezado y gruesa abajo, sin rejilla), Times New Roman de 10,5 pt,
leyenda «Cuadro N: …» encima con el estilo de descripción del documento y la fuente como nota al pie.
"""

import copy
import re

from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TNR = ('<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/>')


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs(texto, base_rpr="", tnr=False):
    """Convierte «**negrita**» y «*cursiva*» en runs de Word."""
    salida = []
    for parte in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", texto):
        if not parte:
            continue
        rpr = (TNR if tnr else "") + base_rpr
        if parte.startswith("**"):
            rpr += "<w:b/><w:bCs/>"
            parte = parte[2:-2]
        elif parte.startswith("*"):
            rpr += "<w:i/><w:iCs/>"
            parte = parte[1:-1]
        rpr += '<w:sz w:val="21"/>' if tnr else ""
        salida.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{_esc(parte)}</w:t></w:r>')
    return "".join(salida)


class Guion:
    """Agrega contenido al final del cuerpo de un ``docx.Document`` (antes de ``sectPr``)."""

    def __init__(self, doc):
        self.doc = doc
        self.body = doc.element.body
        self.cuadro = self._ultimo_cuadro()
        self.fn_part = next(r.target_part for r in doc.part.rels.values() if r.reltype.endswith("/footnotes"))
        self.fn_xml = etree.fromstring(self.fn_part.blob)
        # Notas que ya no tienen referencia en el documento (por ejemplo, de una versión anterior de la sección)
        usadas = {r.get(qn("w:id")) for r in self.body.iter(qn("w:footnoteReference"))}
        for f in self.fn_xml.findall(qn("w:footnote")):
            if int(f.get(qn("w:id"))) > 0 and f.get(qn("w:id")) not in usadas:
                self.fn_xml.remove(f)
        self.fn_id = max(int(f.get(qn("w:id"))) for f in self.fn_xml.findall(qn("w:footnote")))

    def _ultimo_cuadro(self):
        nums = [int(m) for p in self.doc.paragraphs for m in re.findall(r"^Cuadro (\d+):", p.text)]
        return max(nums) if nums else 0

    ancla = None  # si se fija, el contenido se inserta después de este elemento y no al final

    def desde(self, elemento):
        """Inserta lo que siga justo después de ``elemento`` (por ejemplo, el recuadro de un criterio)."""
        self.ancla = elemento

    def _agregar(self, xml):
        el = parse_xml(xml)
        if self.ancla is not None:
            self.ancla.addnext(el)
            self.ancla = el
        else:
            self.body.insert(len(self.body) - 1, el)  # antes de sectPr
        return el

    def titulo(self, texto, nivel=1, pagina_nueva=False):
        estilo = self.doc.styles[f"Heading {nivel}"].style_id
        salto = "<w:pageBreakBefore/>" if pagina_nueva else ""
        self._agregar(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="{estilo}"/>{salto}</w:pPr>{_runs(texto)}</w:p>')

    def parrafo(self, texto):
        self._agregar(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="140" w:line="300" w:lineRule="auto"/>'
                      f'<w:jc w:val="both"/></w:pPr>{_runs(texto)}</w:p>')

    def _nota(self, texto):
        self.fn_id += 1
        nota = parse_xml(f'<w:footnote {nsdecls("w")} w:id="{self.fn_id}"><w:p><w:pPr><w:pStyle w:val="Textonotapie"/>'
                         '</w:pPr><w:r><w:rPr><w:rStyle w:val="Refdenotaalpie"/></w:rPr><w:footnoteRef/></w:r>'
                         f'<w:r><w:t xml:space="preserve"> {_esc(texto)}</w:t></w:r></w:p></w:footnote>')
        self.fn_xml.append(nota)
        return self.fn_id

    def cuadro_(self, leyenda, encabezado, filas, nota, anchos=None):
        """Leyenda «Cuadro N: …» con nota al pie y cuadro de tres reglas; la primera columna va alineada a la izquierda."""
        self.cuadro += 1
        fid = self._nota(nota)
        self._agregar(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="Descripcin"/><w:keepNext/><w:spacing w:before="200"/>'
                      f'</w:pPr>{_runs(f"Cuadro {self.cuadro}: {leyenda}", tnr=True)}<w:r><w:rPr>'
                      f'<w:rStyle w:val="Refdenotaalpie"/></w:rPr><w:footnoteReference w:id="{fid}"/></w:r></w:p>')
        n = len(encabezado)
        anchos = anchos or [9360 // n] * n
        grid = "".join(f'<w:gridCol w:w="{a}"/>' for a in anchos)

        def celda(texto, ancho, arriba, abajo, izquierda, seguir=True):
            b_arr = f'<w:top w:val="single" w:sz="{arriba}" w:space="0" w:color="000000"/>' if arriba else '<w:top w:val="nil"/>'
            b_aba = f'<w:bottom w:val="single" w:sz="{abajo}" w:space="0" w:color="000000"/>' if abajo else '<w:bottom w:val="nil"/>'
            jc = "" if izquierda else '<w:jc w:val="center"/>'
            return (f'<w:tc><w:tcPr><w:tcW w:w="{ancho}" w:type="dxa"/><w:tcBorders>{b_arr}<w:left w:val="nil"/>{b_aba}'
                    f'<w:right w:val="nil"/></w:tcBorders></w:tcPr><w:p><w:pPr>{"<w:keepNext/>" if seguir else ""}'
                    f'<w:spacing w:before="20" w:after="20"/>{jc}</w:pPr>{_runs(str(texto), tnr=True)}</w:p></w:tc>')

        filas_xml = ['<w:tr><w:trPr><w:tblHeader/><w:jc w:val="center"/></w:trPr>' +
                     "".join(celda(h, a, 12, 6, j == 0) for j, (h, a) in enumerate(zip(encabezado, anchos))) + "</w:tr>"]
        for i, fila in enumerate(filas):
            ultima = i == len(filas) - 1
            filas_xml.append('<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>' + "".join(
                celda(v, a, 0, 12 if ultima else 0, j == 0, not ultima) for j, (v, a) in enumerate(zip(fila, anchos)))
                + "</w:tr>")
        self._agregar(f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/>'
                      '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
                      '<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLook w:val="04A0" '
                      'w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
                      f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(filas_xml)}</w:tbl>')
        self._agregar(f'<w:p {nsdecls("w")}><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>')

    def guardar(self, ruta):
        self.fn_part._blob = etree.tostring(self.fn_xml, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.doc.save(ruta)


def quitar_seccion(doc, titulo):
    """Elimina una sección (desde su título de nivel 1 hasta el siguiente título de nivel 1 o el final)."""
    body = doc.element.body
    estilo = doc.styles["Heading 1"].style_id
    dentro = False
    for el in list(body.iterchildren()):
        if el.tag == qn("w:sectPr"):
            break
        es_h1 = el.tag == qn("w:p") and el.find(f".//{qn('w:pStyle')}") is not None and \
            el.find(f".//{qn('w:pStyle')}").get(qn("w:val")) == estilo
        texto = "".join(t.text or "" for t in el.iter(qn("w:t")))
        if es_h1:
            dentro = texto.strip().startswith(titulo)
        if dentro:
            body.remove(el)
    return doc


__all__ = ["Guion", "quitar_seccion", "copy"]
