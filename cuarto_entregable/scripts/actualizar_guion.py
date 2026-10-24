"""Agrega (o reemplaza) la sección «8. Mejoras y ajustes del modelo» del guion de sustentación.

Todas las cifras salen de los resultados guardados por el cuaderno y por los scripts de mejoras, de modo que el guion y
el cuaderno coinciden. Ejecutar después de ``scripts/construir_cuaderno.py --ejecutar``.

Uso (desde la raíz del proyecto):
    python scripts/actualizar_guion.py
"""

import json
import pickle
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import docx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

from bm import config as cfg  # noqa: E402
from bm.anidada import particion  # noqa: E402
from bm.mejoras import UMBRAL_SEMILLAS, tabla_arboles, tabla_mejoras  # noqa: E402
from guion_docx import Guion, quitar_seccion  # noqa: E402

GUION = RAIZ / "Guion de sustentación - Rúbrica - Bank Marketing.docx"
CUADERNO = "«Proyecto Final - Cuarto Entregable - Bank Marketing.ipynb»"
TITULO = "8. Mejoras y ajustes del modelo"
NOMBRE = {"naive_bayes": "Naive Bayes", "logistica": "Logística L1/L2", "bosque": "Random Forest", "svm": "SVM",
          "xgboost": "XGBoost", "arbol": "Árbol de decisión", "knn": "KNN", "referencia_e2": "Referencia E2"}


def num(x, d=3):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pval(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=None, help="ruta de salida; por defecto, sobrescribe el guion")
    salida = Path(ap.parse_args().salida or GUION)
    res = cfg.RUTA_RESULTADOS
    p = particion()
    yte = p.y_te.to_numpy()
    r3 = json.loads((res / "mejoras" / "resumen_r3.json").read_text(encoding="utf-8"))
    flex = pd.read_parquet(res / "mejoras" / "flexibles.parquet")
    rango = pd.read_parquet(res / "mejoras" / "rango_macro.parquet").set_index("Variable")
    arb = tabla_arboles().set_index("variante")
    mej = tabla_mejoras().set_index("clave")
    with open(res / "e3" / "finales_clf.pkl", "rb") as f:
        fin_e3 = pickle.load(f)
    with open(res / "finales_original" / "l2e.pkl", "rb") as f:
        l2e = pickle.load(f)

    doc = quitar_seccion(docx.Document(GUION), TITULO)
    g = Guion(doc)
    g.titulo(TITULO, 1, pagina_nueva=True)
    g.parrafo("Esta sección documenta, en orden, lo que se hizo después del tercer entregable para mejorar los resultados. "
              "Todos los cambios siguieron una misma regla, declarada antes de ejecutarlos: cada uno pasa por la misma "
              "validación anidada (mismos pliegues, Optuna y 20 configuraciones) y **solo se adopta si su AUC-PR medio del "
              f"bucle externo supera al del modelo vigente en más de {num(UMBRAL_SEMILLAS, 3)}**, que es lo que varía el "
              "resultado al cambiar la semilla del optimizador. La prueba se reporta, pero no decide, y se presentan también "
              "los cambios que no funcionaron.")

    # 8.1 Diagnóstico
    g.titulo("8.1 Diagnóstico: por qué los modelos flexibles no superaban a la logística", 2)
    lo, rf, xg = flex.loc["Logística C = 0,01"], flex.loc["Random Forest"], flex.loc["XGBoost"]
    fuera = rango.loc["nr.employed", "Prueba fuera del rango de entrenamiento"]
    g.parrafo("Antes de mejorar había que descartar un error. Si el código estuviera mal, los modelos flexibles perderían "
              "en cualquier escenario; sin embargo, con validación aleatoria dentro del entrenamiento, donde no hay cambio "
              f"de régimen, Random Forest y XGBoost **sí superan** a la logística ({num(rf['AUC-PR aleatoria'])} y "
              f"{num(xg['AUC-PR aleatoria'])} frente a {num(lo['AUC-PR aleatoria'])} de AUC-PR). Pierden en la prueba porque "
              f"el {num(100 * fuera, 1)} % de sus registros tiene un «nr.employed» por debajo del mínimo visto en el "
              "entrenamiento, y un árbol no extrapola: fuera de ese rango su predicción es constante, mientras que la "
              "logística sigue la tendencia. El límite del problema no es la forma funcional del modelo, sino el cambio de "
              "régimen en el tiempo.")
    g.cuadro_("Modelos flexibles frente a la logística según el escenario de validación (AUC-PR)",
              ["Modelo", "Validación aleatoria", "Validación temporal", "Prueba (2009-2010)", "AUC prueba"],
              [[n, num(r["AUC-PR aleatoria"]), num(r["AUC-PR temporal"]), num(r["AUC-PR prueba"]), num(r["AUC prueba"])]
               for n, r in (("Logística C = 0,01", lo), ("Random Forest", rf), ("XGBoost", xg))],
              f"Fuente: cuaderno {CUADERNO}, sección 1.1, subsección «¿Por qué ningún modelo más flexible supera a la "
              "logística?». Validación aleatoria: StratifiedKFold de 5 pliegues dentro del entrenamiento; validación "
              "temporal: los 5 pliegues externos; hiperparámetros fijos para los tres modelos.",
              anchos=[2760, 1650, 1650, 1650, 1650])

    # 8.2 Ajustes para los árboles
    g.titulo("8.2 Ajustes evaluados para los modelos de árboles", 2)
    filas = []
    for v in arb.index:
        r = arb.loc[v]
        dif = "—" if pd.isna(r["diferencia"]) else f"{num(r['diferencia'])} [{num(r['ic_inf'])}; {num(r['ic_sup'])}]"
        filas.append([v, num(r["AUC-PR externo"]) if pd.notna(r["AUC-PR externo"]) else "—", num(r["AUC-PR prueba"]),
                      dif, "—" if pd.isna(r["p"]) else pval(r["p"])])
    rf_a, xg_a = arb.loc["Random Forest · (a) sin macro ni month"], arb.loc["XGBoost · (a) sin macro ni month"]
    rf_e = arb.loc["Random Forest · (e) reentrenado cada mes"]
    g.parrafo("Con ese diagnóstico se probaron cinco ajustes. Quitar las variables macroeconómicas mejora a XGBoost en la "
              f"prueba ({num(xg_a['diferencia'])}, p {('< 0,001' if xg_a['p'] < 0.001 else '= ' + num(xg_a['p']))}), pero en "
              f"Random Forest la diferencia no es significativa (p = {num(rf_a['p'])}) y en el bucle externo empeora a ambos, "
              "así que no se adopta como cambio de los benchmarks. Expresar las macro como cambio mensual no ayuda y el "
              "híbrido sobre la logística apenas la iguala. Lo que sí funciona es usar las etiquetas recientes: con el filtro "
              f"de nivel del modelo original o reentrenando el bosque cada mes, que en la prueba llega a "
              f"{num(rf_e['AUC-PR prueba'], 4)}, prácticamente igual al L2E ({num(arb.loc['L2E (modelo original)', 'AUC-PR prueba'], 4)}), "
              f"pero en el bucle externo se queda en {num(rf_e['AUC-PR externo'])} y exige reentrenar el bosque cada mes.")
    g.cuadro_("Ajustes para que los árboles toleren el cambio de régimen (AUC-PR)",
              ["Ajuste", "Bucle externo", "Prueba", "Δ prueba frente al original [IC 95 %]", "p"], filas,
              f"Fuente: cuaderno {CUADERNO}, sección 6.13, generado con scripts/ejecutar_ajustes_arboles.py. La diferencia "
              "se calcula frente a la versión original del mismo modelo, con intervalo y valor p por bootstrap estacionario "
              "de días (1 000 réplicas). (c) es XGBoost sobre el logit de la logística; (d) usa el filtro de nivel del L2E.",
              anchos=[3360, 1100, 1000, 2700, 1200])

    # 8.3 Modelo original
    g.titulo("8.3 Ajuste principal: el modelo original L2E", 2)
    mejor = r3["mejor"]
    filas = []
    for k, nombre in (("L2E", "L2E (modelo original)"), (mejor, f"{NOMBRE[mejor]} (mejor combinación)"),
                      ("referencia_e2", "Referencia E2 (logística C = 0,01)")):
        s = l2e["puntaje_te"] if k == "L2E" else fin_e3[f"clf_{k}"]["puntaje_te"]
        ext = (f"{num(r3['ext_l2e'])} ± {num(r3['ext_l2e_de'])}" if k == "L2E" else
               f"{num(r3['ext_mejor'])} ± {num(r3['ext_mejor_de'])}" if k == mejor else "—")
        filas.append([nombre, ext, num(average_precision_score(yte, s)), num(roc_auc_score(yte, s)),
                      num(brier_score_loss(yte, s))])
    c2 = r3["control_c2"]
    g.parrafo("La mejora de fondo fue cambiar de enfoque: en lugar de un clasificador más flexible, un modelo que trata el "
              "nivel de conversión de cada día como un estado latente y lo actualiza con las etiquetas de los días cerrados. "
              f"Frente a la mejor combinación del tercer entregable, {r3['n_r3']} de las 4 pruebas declaradas rechazan la "
              f"igualdad, con un δ de Cliff de {num(r3['cliff'])}, y la condición R3 {'se cumple' if r3['r3'] else 'no se cumple'}. "
              f"El mecanismo es transferible: {c2['modelo']} {c2['variante']} llega a {num(c2['auc_pr'])}, a "
              f"{num(c2['dif'])} del L2E (p = {num(c2['p'])}), lo que muestra que el aporte principal es el filtro del nivel.")
    g.cuadro_("Modelo original frente a los benchmarks del tercer entregable",
              ["Modelo", "AUC-PR externo (media ± DE)", "AUC-PR prueba", "AUC prueba", "Brier prueba"], filas,
              f"Fuente: cuaderno {CUADERNO}, secciones 6.2 (tabla resumen), 6.7 (pruebas de hipótesis) y 6.8 (controles con "
              "igual información). El bucle externo de la Referencia E2 no existe porque no forma parte de las 140 "
              "combinaciones.", anchos=[3100, 2060, 1400, 1400, 1400])

    # 8.4 Mejoras del L2E
    g.titulo("8.4 Mejoras evaluadas sobre el modelo original (M1 a M5)", 2)
    filas = [[mej.loc[k, "Mejora"], num(mej.loc[k, "AUC-PR externo"], 4),
              "—" if k == "l2e" else num(mej.loc[k, "Δ externo frente al L2E"], 4), num(mej.loc[k, "Último pliegue"]),
              num(mej.loc[k, "AUC-PR prueba"]), mej.loc[k, "Decisión"]] for k in mej.index]
    adoptadas = mej[mej["Decisión"] == "Se adopta"]
    m2 = mej.loc["m2_bayes_empirico"]
    m4 = mej.loc["m4_ensamble"]
    g.parrafo("Sobre el L2E se probaron cinco mejoras habituales. Los *splines* para la edad y el número de contactos (M1), "
              "la ponderación por recencia (M3) y el promedio de cuatro semillas del optimizador (M5) cambian el AUC-PR "
              "externo menos que el umbral, así que no se distinguen del ruido de la optimización. El ensamble con un bosque "
              f"reentrenado cada mes (M4) es el caso que más cuidado exige: en la prueba sube a {num(m4['AUC-PR prueba'])}, "
              f"por encima del L2E ({num(mej.loc['l2e', 'AUC-PR prueba'])}), pero en el bucle externo baja "
              f"{num(-m4['Δ externo frente al L2E'], 4)}; por la regla no se adopta, porque elegirlo por su resultado en la "
              "prueba sería usar la prueba para seleccionar el modelo, y queda como línea de trabajo futuro que exige "
              f"validación adicional. El resultado más informativo es negativo: estimar q_u por Bayes empírico (M2) "
              f"baja el AUC-PR externo a {num(m2['AUC-PR externo'], 4)}, porque la evidencia aprende la dinámica del nivel "
              "con 2008, cuando casi no se movía, y el filtro queda demasiado lento para el salto de 2009 "
              f"({num(m2['Último pliegue'])} frente a {num(mej.loc['l2e', 'Último pliegue'])} en el último pliegue).")
    g.cuadro_("Mejoras evaluadas sobre el L2E con la misma validación anidada (AUC-PR)",
              ["Variante", "Bucle externo", "Δ frente al L2E", "Último pliegue", "Prueba", "Decisión"], filas,
              f"Fuente: cuaderno {CUADERNO}, sección 6.13, generado con scripts/ejecutar_mejoras.py. Regla de decisión: se "
              f"adopta si el AUC-PR medio externo supera al del L2E en más de {num(UMBRAL_SEMILLAS, 3)} (rango entre las "
              "semillas 42, 1, 2 y 3 del optimizador). La prueba no interviene en la decisión.",
              anchos=[3300, 1150, 1150, 1150, 1000, 1610])

    # 8.5 Decisión
    g.titulo("8.5 Decisión y mensaje para la sustentación", 2)
    g.parrafo(("Por tal motivo, se conserva el L2E sin cambios como modelo final: ninguna de las mejoras supera la regla de "
               "decisión, y las que más ayudan a los modelos de referencia son justamente las que usan las etiquetas "
               "recientes, que es el mecanismo del modelo original. Si el jurado pregunta por M4, la respuesta es que mejora "
               "la prueba pero no la validación anidada, y que adoptarlo con esa evidencia sería seleccionar con la prueba. ")
              if adoptadas.empty else
              (f"Por tal motivo, se adopta {', '.join(adoptadas['Mejora'])}. ") +
              "En la sustentación, el mensaje es directo: el problema no se resolvía con modelos más complejos sino "
              "modelando el tiempo; se probaron diez ajustes con la misma regla y todos se reportan, también los que no "
              "funcionaron, porque mostrar dónde no mejora el modelo es parte del rigor que pide la guía.")
    g.guardar(salida)
    print(f"Sección «{TITULO}» escrita en {salida.name} (cuadros {g.cuadro - 3} a {g.cuadro})")


if __name__ == "__main__":
    main()
