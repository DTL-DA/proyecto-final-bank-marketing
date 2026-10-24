"""Revisión sistemática de literatura para la declaración de novedad del modelo original (guía, sección 5.2).

Consulta cuatro bases bibliográficas con APIs públicas (OpenAlex, arXiv, Crossref y Semantic Scholar) con ocho cadenas
que combinan (i) la familia del modelo, (ii) el problema de aplicación y (iii) sinónimos en inglés y español. Para cada
consulta guarda la fecha, el número total de resultados y los primeros ``N_REVISADOS`` registros (título, año, DOI y
resumen) en ``resultados/literatura/consultas.json``. El tamizaje marca como candidatos los registros cuyo título o
resumen mencionan a la vez el problema (telemercadeo bancario o depósito a término) y un mecanismo temporal del modelo
(estado latente, Kalman, logística dinámica, deriva o cambio de prevalencia); esos candidatos se leen uno por uno y se
clasifican en el cuaderno (sección 4).

Uso (desde la raíz del proyecto):
    python scripts/revision_literatura.py
"""

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "resultados" / "literatura"
N_REVISADOS = 25
AGENTE = {"User-Agent": "revision-literatura-ml-uninorte/1.0 (mailto:anonimo@example.org)"}

# (cadena en sintaxis booleana, términos para APIs de texto libre)
# Las dos primeras son cadenas de control (solo la aplicación y solo la familia del modelo): verifican que la sintaxis
# devuelve resultados y dimensionan cada literatura; las ocho siguientes son las intersecciones que evalúan la novedad.
CONTROLES = 2
CADENAS = [
    ('"bank telemarketing" OR ("bank marketing" AND "term deposit")', "bank telemarketing term deposit"),
    ('"dynamic logistic regression" OR ("state space" AND "logistic regression" AND "time-varying")',
     "dynamic logistic regression state space time-varying"),
    ('"bank telemarketing" AND ("state space" OR "Kalman")', "bank telemarketing state space Kalman filter"),
    ('"bank marketing" AND "dynamic logistic regression"', "bank marketing dynamic logistic regression"),
    ('"term deposit" AND ("concept drift" OR "prior shift" OR "label shift")',
     "term deposit subscription concept drift prior shift"),
    ('"bank telemarketing" AND ("time-varying" OR "random walk") AND "logistic"',
     "bank telemarketing time-varying random walk logistic"),
    ('"bank marketing" AND ("online learning" OR "delayed labels" OR "data stream")',
     "bank marketing online learning delayed labels data stream"),
    ('("telemercadeo" OR "telemarketing") AND ("depósito a plazo" OR "depósito a término") AND ("Kalman" OR "modelo dinámico")',
     "telemercadeo depósito a plazo modelo dinámico filtro de Kalman"),
    ('"direct marketing" AND "dynamic logistic regression"', "direct marketing dynamic logistic regression"),
    ('"dynamic logistic regression" AND ("delayed labels" OR "label delay" OR "verification latency")',
     "dynamic logistic regression delayed labels verification latency"),
]

PROBLEMA = re.compile(r"bank(ing)? (tele)?marketing|telemarketing|term deposit|dep[oó]sito a (plazo|t[eé]rmino)|"
                      r"telemercadeo", re.I)
MECANISMO = re.compile(r"state[- ]space|kalman|dynamic logistic|time[- ]varying|random walk|concept drift|"
                       r"prior shift|label shift|data stream|online learning|delayed label|latent|modelo din[aá]mico|"
                       r"deriva", re.I)


def _json(url, reintentos=7):
    for i in range(reintentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < reintentos - 1:
                time.sleep(10 * 2 ** min(i, 3))  # espera creciente: 10, 20, 40 y 80 s
                continue
            raise
    return None


def _resumen_openalex(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def openalex(booleana, libre):
    """OpenAlex: búsqueda booleana en título y resumen (``title_and_abstract.search``)."""
    q = urllib.parse.quote(booleana)
    d = _json(f"https://api.openalex.org/works?filter=title_and_abstract.search:{q}&per-page={N_REVISADOS}"
              "&select=id,doi,title,publication_year,abstract_inverted_index,primary_location")
    regs = [{"titulo": w.get("title") or "", "anio": w.get("publication_year"), "doi": w.get("doi"),
             "resumen": _resumen_openalex(w.get("abstract_inverted_index")),
             "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}
            for w in d["results"]]
    return d["meta"]["count"], regs


def arxiv(booleana, libre):
    """arXiv: búsqueda booleana en título y resumen (prefijo ``all:`` sobre cada término entre comillas)."""
    consulta = re.sub(r'"([^"]+)"', lambda m: 'all:"' + m.group(1) + '"', booleana)
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(consulta) +
           f"&start=0&max_results={N_REVISADOS}")
    for intento in range(5):  # arXiv responde 429 si se consulta seguido: espera creciente de 60 a 240 s
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60) as r:
                raiz = ET.fromstring(r.read())
            break
        except (urllib.error.HTTPError, TimeoutError) as e:
            if intento == 4 or (isinstance(e, urllib.error.HTTPError) and e.code not in (429, 503)):
                raise
            time.sleep(60 * min(2 ** intento, 4))
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(raiz.find("o:totalResults", ns).text)
    regs = [{"titulo": " ".join(e.find("a:title", ns).text.split()), "anio": int(e.find("a:published", ns).text[:4]),
             "doi": e.find("a:id", ns).text, "resumen": " ".join(e.find("a:summary", ns).text.split()), "fuente": "arXiv"}
            for e in raiz.findall("a:entry", ns)]
    return total, regs


def crossref(booleana, libre):
    """Crossref: búsqueda bibliográfica por relevancia (sin operadores booleanos); se revisan los primeros registros."""
    d = _json("https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(libre) +
              f"&rows={N_REVISADOS}&select=DOI,title,issued,abstract,container-title")
    regs = [{"titulo": (w.get("title") or [""])[0], "anio": (w.get("issued", {}).get("date-parts") or [[None]])[0][0],
             "doi": "https://doi.org/" + w["DOI"], "resumen": re.sub(r"<[^>]+>", " ", w.get("abstract") or ""),
             "fuente": (w.get("container-title") or [""])[0]} for w in d["message"]["items"]]
    return d["message"]["total-results"], regs


def semantic_scholar(booleana, libre):
    """Semantic Scholar: búsqueda por relevancia en título y resumen."""
    d = _json("https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(libre) +
              f"&limit={N_REVISADOS}&fields=title,year,abstract,externalIds,venue")
    regs = [{"titulo": w.get("title") or "", "anio": w.get("year"),
             "doi": (w.get("externalIds") or {}).get("DOI"), "resumen": w.get("abstract") or "",
             "fuente": w.get("venue")} for w in d.get("data", [])]
    return d.get("total", 0), regs


BASES = {"OpenAlex": openalex, "arXiv": arxiv, "Crossref": crossref, "Semantic Scholar": semantic_scholar}


def tamizar(r):
    """Candidato si el título o el resumen mencionan el problema y un mecanismo temporal a la vez."""
    texto = f"{r['titulo']} {r['resumen']}"
    return bool(PROBLEMA.search(texto)) and bool(MECANISMO.search(texto))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="*", default=None, help="repite solo estas bases y conserva las demás consultas")
    args = ap.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    previas = []
    if args.bases:
        previas = [q for q in json.loads((DESTINO / "consultas.json").read_text(encoding="utf-8"))
                   if q["base"] not in args.bases]
    consultas = []
    for base, f in BASES.items():
        if args.bases and base not in args.bases:
            continue
        for booleana, libre in CADENAS:
            fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                total, regs = f(booleana, libre)
                error = None
            except Exception as e:  # se registra el fallo para no ocultar consultas que no se pudieron hacer
                total, regs, error = None, [], f"{type(e).__name__}: {e}"
            for r in regs:
                r["candidato"] = tamizar(r)
            consultas.append({"base": base, "control": CADENAS.index((booleana, libre)) < CONTROLES,
                              "cadena": booleana if base in ("OpenAlex", "arXiv") else libre,
                              "fecha_utc": fecha, "resultados": total, "revisados": len(regs),
                              "candidatos": int(sum(r["candidato"] for r in regs)), "error": error, "registros": regs})
            print(f"{base:17s} {total!s:>9} res. · {len(regs):2d} rev. · "
                  f"{sum(r['candidato'] for r in regs):2d} cand. · {booleana[:70]}", flush=True)
            time.sleep({"arXiv": 10, "Semantic Scholar": 12}.get(base, 1.5))  # arXiv pide al menos 3 s entre consultas
    # Segundo intento de las consultas fallidas, tras una pausa larga (límites de peticiones de las APIs públicas)
    fallidas = [q for q in consultas if q["error"]]
    if fallidas:
        time.sleep(90)
    for q in fallidas:
        booleana, libre = next(c for c in CADENAS if q["cadena"] in c)
        try:
            total, regs = BASES[q["base"]](booleana, libre)
            for r in regs:
                r["candidato"] = tamizar(r)
            q.update({"resultados": total, "revisados": len(regs), "candidatos": int(sum(r["candidato"] for r in regs)),
                      "error": None, "registros": regs, "reintento": True,
                      "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            print(f"Reintento correcto: {q['base']} · {q['cadena'][:60]} · {total} res.", flush=True)
        except Exception as e:
            q["error"] = f"{type(e).__name__}: {e} (también en el reintento)"
        time.sleep(30 if q["base"] == "Semantic Scholar" else 4)
    orden = list(BASES)
    consultas = sorted(previas + consultas, key=lambda q: orden.index(q["base"]))
    (DESTINO / "consultas.json").write_text(json.dumps(consultas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Consultas guardadas en {DESTINO / 'consultas.json'}")


if __name__ == "__main__":
    main()
