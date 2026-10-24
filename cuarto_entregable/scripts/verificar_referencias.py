"""Verifica las referencias por DOI y genera ``resultados/literatura/references.bib``.

Para cada DOI pide la entrada BibTeX a doi.org (negociación de contenido), que la construye con los metadatos que
registró la editorial; así título, autores, año y revista no dependen de transcripciones manuales. Las referencias sin
DOI (actas de JMLR, NeurIPS, ICML, informes técnicos) se escriben a mano y se marcan como tales en el archivo.

Uso (desde la raíz del proyecto):
    python scripts/verificar_referencias.py
"""

import json
import re
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "resultados" / "literatura"

DOIS = {
    "moro2014": "10.1016/j.dss.2014.03.001",
    "uci_bank": "10.24432/C5K306",
    "mccormick2012": "10.1111/j.1541-0420.2011.01645.x",
    "fahrmeir1992": "10.1080/01621459.1992.10475232",
    "west1985": "10.1080/01621459.1985.10477131",
    "saerens2002": "10.1162/089976602753284446",
    "gama2014": "10.1145/2523813",
    "grzenda2020": "10.1007/s10618-019-00654-y",
    "mackay1992": "10.1162/neco.1992.4.5.720",
    "durbin2012": "10.1093/acprof:oso/9780199641178.001.0001",
    "chowdary2025": "10.1109/icscsa66339.2025.11170856",
    "kalman1960": "10.1115/1.3662552",
    "delong1988": "10.2307/2531595",
    "nadeau2003": "10.1023/A:1024068626366",
    "diebold1995": "10.1080/07350015.1995.10524599",
    "harvey1997": "10.1016/S0169-2070(96)00719-4",
    "politis1994": "10.1080/01621459.1994.10476870",
    "akiba2019": "10.1145/3292500.3330701",
    "cliff1993": "10.1037/0033-2909.114.3.494",
    "zhang2008": "10.1007/s10115-007-0095-1",
    "boyd2004": "10.1017/CBO9780511804441",
    "varma2006": "10.1186/1471-2105-7-91",
    "cawley2010": None,
}

MANUALES = r"""
@article{demsar2006,
  author = {Dem{\v{s}}ar, Janez}, title = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research}, volume = {7}, pages = {1--30}, year = {2006}, note = {Sin DOI}}
@article{benavoli2017,
  author = {Benavoli, Alessio and Corani, Giorgio and Dem{\v{s}}ar, Janez and Zaffalon, Marco},
  title = {Time for a Change: a Tutorial for Comparing Multiple Classifiers Through {B}ayesian Analysis},
  journal = {Journal of Machine Learning Research}, volume = {18}, number = {77}, pages = {1--36}, year = {2017},
  note = {Sin DOI}}
@article{cawley2010,
  author = {Cawley, Gavin C. and Talbot, Nicola L. C.},
  title = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  journal = {Journal of Machine Learning Research}, volume = {11}, pages = {2079--2107}, year = {2010}, note = {Sin DOI}}
@inproceedings{lipton2018,
  author = {Lipton, Zachary C. and Wang, Yu-Xiang and Smola, Alexander J.},
  title = {Detecting and Correcting for Label Shift with Black Box Predictors},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning}, series = {PMLR},
  volume = {80}, pages = {3122--3130}, year = {2018}, note = {arXiv:1802.03916}}
@inproceedings{garg2020,
  author = {Garg, Saurabh and Wu, Yifan and Balakrishnan, Sivaraman and Lipton, Zachary C.},
  title = {A Unified View of Label Shift Estimation},
  booktitle = {Advances in Neural Information Processing Systems}, volume = {33}, year = {2020},
  note = {arXiv:2003.07554}}
@inproceedings{lundberg2017,
  author = {Lundberg, Scott M. and Lee, Su-In}, title = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems}, volume = {30}, year = {2017},
  note = {arXiv:1705.07874}}
@inproceedings{bergstra2011,
  author = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  title = {Algorithms for Hyper-Parameter Optimization},
  booktitle = {Advances in Neural Information Processing Systems}, volume = {24}, year = {2011}}
@inproceedings{romano2006,
  author = {Romano, Jeanine and Kromrey, Jeffrey D. and Coraggio, Jesse and Skowronek, Jeff},
  title = {Appropriate Statistics for Ordinal Level Data: Should We Really Be Using t-test and {C}ohen's d for
           Evaluating Group Differences on the {NSSE} and Other Surveys?},
  booktitle = {Annual Meeting of the Florida Association of Institutional Research}, year = {2006}}
@article{holm1979,
  author = {Holm, Sture}, title = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics}, volume = {6}, number = {2}, pages = {65--70}, year = {1979}}
@techreport{harries1999,
  author = {Harries, Michael}, title = {{SPLICE-2} Comparative Evaluation: Electricity Pricing},
  institution = {University of New South Wales}, year = {1999}}
@misc{ikonomovska2009,
  author = {Ikonomovska, Elena}, title = {Airline Dataset (Data Expo 2009, versión para flujos de datos)},
  howpublished = {OpenML, conjunto 1169}, year = {2009}}
"""


# doi.org registra la fecha de publicación en línea; estas obras se citan por el año del volumen impreso
ANIO_VOLUMEN = {"mccormick2012": 2012, "zhang2008": 2008, "grzenda2020": 2020}


def bibtex(doi: str, clave: str) -> str:
    """Entrada BibTeX desde doi.org, con la clave del proyecto."""
    req = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8",
                                                                    "User-Agent": "verificar-referencias/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        texto = r.read().decode("utf-8").strip()
    texto = re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{clave},", texto, count=1)
    if clave in ANIO_VOLUMEN:
        texto = re.sub(r"year=\{\d{4}\}", f"year={{{ANIO_VOLUMEN[clave]}}}", texto)
    return texto


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    entradas, informe = [], []
    for clave, doi in DOIS.items():
        if doi is None:
            continue
        try:
            entradas.append(bibtex(doi, clave))
            informe.append({"clave": clave, "doi": doi, "verificada": True})
        except Exception as e:
            informe.append({"clave": clave, "doi": doi, "verificada": False, "error": str(e)})
        time.sleep(0.5)
    texto = "% Generado por scripts/verificar_referencias.py: entradas con DOI verificadas en doi.org\n\n"
    texto += "\n\n".join(entradas) + "\n\n% Referencias sin DOI (transcritas a mano)\n" + MANUALES
    (DESTINO / "references.bib").write_text(texto, encoding="utf-8")
    (DESTINO / "verificacion_referencias.json").write_text(json.dumps(informe, indent=1), encoding="utf-8")
    for r in informe:
        print(("OK " if r["verificada"] else "ERR"), r["clave"], r["doi"], r.get("error", ""))


if __name__ == "__main__":
    main()
