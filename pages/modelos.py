"""Sección 3 · Modelos: tabla de líderes, predicción en vivo con SHAP local, diagnóstico, SHAP global, comparación
estadística y subsección del modelo original (L2E)."""
import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from src import plots
from src.data import (banco, contexto, esquema, fila_entrada, fondo, modelos, opciones_modelo, percentil, predecir,
                      probabilidad, puntaje_crudo, resultado, resumen)
from src.explicaciones import shap_permutaciones

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title="Modelos · Bank Marketing")

VARIABLES = [v["variable"] for v in esquema()["variables"]]
ETIQUETAS = {v["variable"]: v["etiqueta"] for v in esquema()["variables"]}
GRUPOS = ["Cliente", "Contacto", "Historial", "Contexto económico"]


def num(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def a_python(v):
    return v.item() if hasattr(v, "item") else v


def tarjeta(titulo, subtitulo, *hijos, id_=None):
    cabecera = [html.H2(titulo, className="h5 card-title mb-1")]
    if subtitulo:
        cabecera.append(html.P(subtitulo, className="text-muted small mb-3"))
    props = {"id": id_} if id_ else {}
    return dbc.Card(dbc.CardBody([*cabecera, *hijos]), className="seccion mb-4", **props)


def grafica(id_grafica, altura):
    # Altura explícita (nunca style=None: dcc.Graph lee style.height y un style nulo reinicia la página en bucle)
    return dcc.Loading(dcc.Graph(id=id_grafica, config=plots.CONFIG, responsive=True, style={"height": f"{altura}px"}),
                       type="dot", color=plots.AZUL)


def pv(p) -> str:
    """Valor p con coma decimal; por debajo de 0,001 se reporta como cota."""
    return "<\u00a00,001" if p < 0.001 else num(p, 3)


def tabla_simple(df, formatos=None, clase=""):
    """Tabla de Bootstrap a partir de un DataFrame; ``formatos`` asigna a cada columna una función de formato."""
    formatos = formatos or {}
    cuerpo = [html.Tr([html.Td(formatos.get(c, str)(r[c])) for c in df.columns]) for _, r in df.iterrows()]
    return dbc.Table([html.Thead(html.Tr([html.Th(c) for c in df.columns])), html.Tbody(cuerpo)], size="sm",
                     striped=True, hover=True, responsive=True, className=f"mb-0 {clase}")


def valor_legible(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}".replace(".", ",")
    return str(v)


def tabla_lideres():
    lb = resultado("leaderboard")
    f3 = Format(precision=3, scheme=Scheme.fixed, decimal_delimiter=",")
    f1 = Format(precision=1, scheme=Scheme.fixed, decimal_delimiter=",")
    columnas = [{"name": "Modelo", "id": "nombre"}, {"name": "Balanceo", "id": "balanceo"},
                {"name": "Optimizador", "id": "optimizador"},
                {"name": "AUC-PR externo (media)", "id": "auc_pr_media", "type": "numeric", "format": f3},
                {"name": "AUC-PR externo (DE)", "id": "auc_pr_de", "type": "numeric", "format": f3},
                {"name": "AUC externo (media)", "id": "auc_media", "type": "numeric", "format": f3},
                {"name": "AUC externo (DE)", "id": "auc_de", "type": "numeric", "format": f3},
                {"name": "Recall externo (media)", "id": "recall_media", "type": "numeric", "format": f3},
                {"name": "AUC-PR en prueba", "id": "auc_pr_prueba", "type": "numeric", "format": f3},
                {"name": "Inferencia (ms por 1 000 clientes)", "id": "inferencia_ms_mil", "type": "numeric", "format": f1},
                {"name": "Combinaciones evaluadas", "id": "combinaciones", "type": "numeric"}]
    return dash_table.DataTable(
        id="mod-tabla", data=lb.to_dict("records"), columns=columnas, sort_action="native", filter_action="native",
        filter_options={"placeholder_text": "Filtrar…", "case": "insensitive"}, page_action="none",
        style_table={"overflowX": "auto"}, style_as_list_view=True,
        style_header={"fontWeight": "700", "backgroundColor": "#F4F5F7", "whiteSpace": "normal", "height": "auto",
                      "color": plots.TINTA},
        style_cell={"fontFamily": "Lato, 'Segoe UI', Arial, sans-serif", "fontSize": 14, "padding": "8px 10px",
                    "textAlign": "left", "minWidth": 90, "whiteSpace": "normal", "height": "auto", "color": plots.TINTA},
        style_cell_conditional=[{"if": {"column_type": "numeric"}, "textAlign": "right"}],
        style_data_conditional=[{"if": {"row_index": 0}, "backgroundColor": "#EAF2FC", "fontWeight": "700"}],
        tooltip_header={"auc_pr_media": "Media de los 5 pliegues externos de la validación anidada",
                        "inferencia_ms_mil": "Tiempo de predicción del artefacto final, medido sobre la prueba"},
    )


def campo(v):
    id_ = {"tipo": "entrada", "var": v["variable"]}
    if v["tipo"] == "categorica":
        control = dcc.Dropdown(id=id_, options=v["valores"], value=v["defecto"], clearable=False,
                               labels=plots.ETIQUETAS_DROPDOWN)
        ayuda = v["ayuda"]
    else:
        especiales = v.get("especiales", [])
        control = dbc.Input(id=id_, type="number", value=v["defecto"], step=v["paso"], debounce=True,
                            min=v["min"], max=max([v["max"], *especiales]))
        rango = f"Entre {num(v['min'], 0 if v['entero'] else 2)} y {num(v['max'], 0 if v['entero'] else 2)}"
        ayuda = rango + (f"; {v['ayuda']}" if v["ayuda"] else "")
    # Dash convierte los id de patrón en un JSON con llaves ordenadas; la etiqueta debe apuntar a ese id
    return html.Div([dbc.Label(v["etiqueta"], html_for=json.dumps(id_, sort_keys=True, separators=(",", ":")),
                               className="mb-1"), control, dbc.FormText(ayuda)], className="mb-3")


def formulario():
    columnas = []
    for g in GRUPOS:
        campos = [campo(v) for v in esquema()["variables"] if v["grupo"] == g]
        columnas.append(dbc.Col(html.Fieldset([html.Legend(g, className="h6"), *campos]), md=6, xl=3))
    return dbc.Row(columnas, className="g-3")


BLOQUES_CD = {"pliegues": "5 pliegues externos de la validación anidada (8 modelos)",
              "meses_prueba": "17 meses de la prueba (9 modelos, con la Referencia E2)"}


def friedman_texto():
    fr = resumen("l2e_resumen")["friedman"]
    pl, me = fr["pliegues"], fr["meses_prueba"]
    nemenyi = resultado("l2e_pruebas").iloc[0]["p crudo"]
    return (f"Con el modelo original incluido, en los {pl['n']} pliegues externos el L2E tiene el mejor rango promedio "
            f"({num(pl['rango_l2e'], 2)}), pero Friedman no rechaza la igualdad (χ² = {num(pl['chi2'], 2)}, "
            f"p = {pv(pl['p'])}): con cinco bloques la diferencia crítica es de {num(pl['cd'], 2)} rangos. En los "
            f"{me['n']} meses de la prueba Friedman sí rechaza (χ² = {num(me['chi2'], 2)}, p {pv(me['p'])}), pero el L2E "
            f"queda {me['puesto_l2e']}.º y Nemenyi no lo separa de Naive Bayes (p = {pv(nemenyi)}). Dentro de un mes el "
            "nivel de conversión cambia poco, y la ventaja del L2E está en seguirlo entre días; por eso su superioridad "
            "se verifica con las pruebas pareadas sobre el ordenamiento global de la prueba (subsección del modelo "
            "original, abajo).")


def modelo_original():
    """Subsección del modelo original: idea, pseudocódigo, literatura, convergencia, pruebas y veredicto."""
    r = resumen("l2e_resumen")
    r3, rob, lit, sp = r["r3"], r["robustez"], r["literatura"], r["ablacion_sp"]
    hp = r["hiperparametros"]
    cifras = [(f"{num(r3['ext_l2e'], 3)} frente a {num(r3['ext_mejor'], 3)}",
               "AUC-PR del bucle externo frente a la mejor combinación"),
              (f"{num(r['auc_pr_prueba'], 3)} frente a {num(r['referencia_e2']['auc_pr_prueba'], 3)}",
               "AUC-PR de prueba frente a la Referencia E2"),
              (f"{r3['n_r3']} de 4", "pruebas declaradas con p de Holm < 0,05"),
              (f"δ = {num(r3['cliff'], 3)}", "tamaño del efecto de Cliff (pequeño)")]
    mantisa, exponente = f"{hp['q_pendiente']:.2e}".split("e")
    q_b = f"{mantisa.replace('.', ',')} · 10{str(int(exponente)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
    idea = dcc.Markdown(
        "El **L2E** (logística de estado latente en dos escalas) combina en el logit un **nivel diario de conversión**, "
        "que se mueve rápido y se actualiza con las etiquetas de los días ya cerrados, y el **efecto de las variables del "
        "cliente**, cuyas pendientes derivan despacio:\n\n"
        r"$$P(y_i = 1) = \sigma\left(u_{t(i)} + x_i^\top \beta_{t(i)}\right)$$" "\n\n"
        r"$$u_t = u_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, q_u)$$" "\n\n"
        r"$$\beta_t = \beta_{t-1} + \nu_t, \qquad \nu_t \sim \mathcal{N}(0,\, q_\beta I)$$" "\n\n"
        "El ajuste maximiza una log-posterior estrictamente cóncava con Newton por bloques (el nivel se elimina con el "
        "complemento de Schur sobre una matriz tridiagonal, en tiempo lineal en el número de días), y la predicción usa un "
        "filtro de Laplace con un día de demora en las etiquetas. Está programado desde cero con numpy y scipy en "
        f"`src/original_model.py`, con {r['pruebas_unitarias']['total']} pruebas unitarias. Hiperparámetros elegidos por "
        rf"Optuna en la validación anidada: $\lambda$ = {num(hp['lam'], 1)}, $q_u$ = {num(hp['q_nivel'], 3)} y "
        rf"$q_\beta$ = {q_b}.",
        mathjax=True, className="mb-0")
    pruebas = resultado("l2e_pruebas")[["Prueba", "H0", "Supuestos verificados", "Estadístico", "p crudo", "p ajustado",
                                         "Efecto", "Conclusión"]]
    conteo = resultado("l2e_conteo")
    ablacion = resultado("l2e_ablacion")[["Variante", "AUC-PR externo", "DE externo", "AUC-PR prueba", "Brier prueba"]]
    f3 = lambda x: num(x, 3)  # noqa: E731
    prueba = resultado("l2e_prueba")
    l2e, bms = prueba.iloc[0], prueba.iloc[1:]
    nb = prueba.set_index("clave").loc["naive_bayes"]
    sin_solape = bms.loc[bms["_sup"] < l2e["_inf"], "Modelo"].str.split(" · ").str[0].tolist()
    tabla_prueba = prueba[["Modelo", "AUC-PR prueba", "IC 95 % AUC-PR", "AUC prueba", "IC 95 % AUC", "Brier prueba"]]
    sig = r3["sig_meses"]
    veredicto = dbc.Alert([
        html.H3([html.I(className="bi bi-check2-circle me-2 text-success"), "Veredicto: superior a los benchmarks"],
                className="h6"),
        html.P([html.B("R1. Novedad: "), f"revisión sistemática en {lit['bases']} bases con {lit['consultas']} consultas y "
                f"{num(lit['revisados'], 0)} registros revisados; ningún estudio aplica este modelo al telemercadeo "
                "bancario, y el trabajo más cercano usa árboles con actualización incremental, otra familia."],
               className="mb-1"),
        html.P([html.B("R2. Implementación propia: "), "clase compatible con scikit-learn (fit, predict, predict_proba) con "
                "la función objetivo, el Newton por bloques y el filtro programados a mano; no es un import."],
               className="mb-1"),
        html.P([html.B("R3. Superioridad: "), f"mayor AUC-PR externo que los 7 finalistas ({num(r3['ext_l2e'], 3)} frente "
                f"a {num(r3['ext_mejor'], 3)}), {r3['n_r3']} de las 4 pruebas declaradas rechazan la igualdad frente a la "
                "mejor combinación con p ajustado por Holm < 0,05, y el efecto es pequeño pero no despreciable "
                f"(δ de Cliff = {num(r3['cliff'], 3)} ≥ 0,147)."], className="mb-0"),
    ], color="light", className="border border-success border-2 mb-3")
    limites = html.Ul([
        html.Li("Friedman y Nemenyi, que comparan rangos dentro de cada mes, no separan al L2E de Naive Bayes."),
        html.Li(f"Dentro de cada mes de prueba es significativamente mejor que {', '.join(sig[:-1])} y {sig[-1]}, pero no "
                "que la logística, Random Forest, la SVM, KNN ni la Referencia E2: su ventaja está en seguir el nivel "
                "entre días, no en ordenar mejor a los clientes de un mismo día."),
        html.Li("En los tres conjuntos de datos adicionales no es el mejor modelo: donde el nivel latente es débil, no "
                "supera a los benchmarks."),
        html.Li("Necesita las etiquetas de los días cerrados: con 40 días de demora el AUC-PR de prueba baja de "
                f"{num(rob['retardo_1'], 3)} a {num(rob['retardo_40'], 3)}."),
        html.Li("En la ablación, la variante sin actualizar las pendientes supera al L2E en el bucle externo por "
                f"{num(sp['dif'], 4)}, una diferencia que no se distingue del ruido (t corregida de Nadeau-Bengio, "
                f"p = {num(sp['p_nb'], 2)}); se conserva el modelo declarado y la prueba no interviene en esa decisión."),
    ], className="mb-0")
    return [
        dbc.Row([dbc.Col(html.Div([html.Div(v, className="cifra-media"), html.Div(tx, className="cifra-texto")],
                                  className="stat stat-claro h-100"), xs=6, lg=3) for v, tx in cifras],
                className="g-3 mb-4"),
        dbc.Row([dbc.Col([html.H3("Idea y formulación", className="h6"), idea], lg=6),
                 dbc.Col([html.H3("Pseudocódigo", className="h6"),
                          html.Pre(r["pseudocodigo"], className="pseudocodigo", tabIndex=0,
                                   **{"aria-label": "Pseudocódigo de los algoritmos de ajuste y de predicción del L2E"})],
                         lg=6)], className="g-4 mb-4"),
        dcc.Graph(figure=plots.nivel_l2e(resultado("l2e_nivel")), config=plots.CONFIG, responsive=True,
                  style={"height": "400px"}),
        html.P("La línea es la probabilidad que el L2E asigna a un cliente promedio cada día; los puntos, la tasa real del "
               "día (su tamaño crece con el número de llamadas). En la prueba el modelo solo conoce los días ya cerrados, y "
               "aun así sigue el salto de 2009, que ningún modelo estático anticipa.", className="insight"),
        dbc.Row([dbc.Col(dcc.Graph(figure=plots.convergencia(resultado("l2e_convergencia")), config=plots.CONFIG,
                                   responsive=True, style={"height": "360px"}), lg=6),
                 dbc.Col(dcc.Graph(figure=plots.aprendizaje(resultado("l2e_aprendizaje")), config=plots.CONFIG,
                                   responsive=True, style={"height": "360px"}), lg=6)], className="g-3"),
        html.P("Optuna encuentra su mejor configuración antes de agotar las 20 evaluaciones del presupuesto, el mismo de los "
               "benchmarks. El L2E logra casi el mismo AUC-PR de prueba con el 25 % de los datos que con todos, mientras "
               "que la logística de referencia necesita más historia: el nivel diario hace que el modelo dependa menos del "
               "tamaño del entrenamiento.", className="insight"),
        html.H3("Diferencias frente a la literatura", className="h6 mt-4"),
        tabla_simple(resultado("l2e_literatura"), clase="tabla-texto"),
        html.H3("Comparación en la prueba frente a los benchmarks", className="h6 mt-4"),
        tabla_simple(tabla_prueba, {"AUC-PR prueba": f3, "AUC prueba": f3,
                                    "Brier prueba": lambda x: "—" if np.isnan(x) else f3(x)}),
        html.P("Prueba de 2009-05 a 2010-11 (8 236 clientes). Intervalos del 95 % por bootstrap estacionario por días "
               "(1 000 réplicas), porque los clientes de un mismo día comparten el nivel. La SVM no tiene Brier porque su "
               "puntaje no es una probabilidad; el diagnóstico del modelo seleccionado la muestra con escalado de Platt.",
               className="nota mt-2"),
        html.P(f"El L2E tiene el mayor AUC-PR ({f3(l2e['AUC-PR prueba'])}) y el mayor AUC ({f3(l2e['AUC prueba'])}) de "
               f"la prueba, y su Brier de {f3(l2e['Brier prueba'])} es el más bajo, frente a {f3(nb['Brier prueba'])} de "
               "Naive Bayes; es decir, además de ordenar mejor a los clientes, sus probabilidades son las más precisas. "
               f"Sin embargo, el intervalo de su AUC-PR, {l2e['IC 95 % AUC-PR']}, se solapa con el de Naive Bayes, "
               f"{nb['IC 95 % AUC-PR']}, y solo se separa del de {len(sin_solape)} de los {len(bms)} benchmarks "
               f"({' y '.join(sin_solape)}). Por tal motivo, la superioridad se verifica con las pruebas pareadas de la "
               "tabla siguiente, que comparan los modelos sobre los mismos días, y no con el solape de los intervalos.",
               className="insight"),
        html.H3("Pruebas de hipótesis frente a la mejor combinación benchmark (Naive Bayes)", className="h6 mt-4"),
        tabla_simple(pruebas, {"p crudo": pv, "p ajustado": pv}, clase="tabla-texto"),
        html.P("p ajustado por Holm sobre las 8 comparaciones del L2E. Las cuatro primeras son las pruebas declaradas para "
               "la condición R3; DeLong supone independencia, que no se cumple porque los clientes de un mismo día "
               "comparten el nivel, y se reporta solo como contraste.", className="nota mt-2"),
        html.H3("Pruebas declaradas frente a cada benchmark", className="h6 mt-4"),
        tabla_simple(conteo, {"δ de Cliff (meses)": f3}),
        html.H3("Estudio de ablación (cada variante reoptimizada con la misma validación anidada)", className="h6 mt-4"),
        tabla_simple(ablacion, {c: f3 for c in ablacion.columns[1:]}),
        html.Div(className="mb-4"),
        veredicto,
        html.H3("Dónde no es superior", className="h6"),
        limites,
    ]


def tabla_delong():
    d = resultado("delong")
    filas = [html.Tr([html.Td(r["Modelo A"]), html.Td(r["Modelo B"]), html.Td(num(r["Diferencia de AUC"], 4)),
                      html.Td(r["IC 95 % BCa"]), html.Td(num(r["p crudo"], 3)), html.Td(num(r["p Holm"], 3))])
             for _, r in d.iterrows()]
    return dbc.Table([html.Thead(html.Tr([html.Th(t) for t in ("Modelo A", "Modelo B", "Diferencia de AUC",
                                                                  "IC 95 % (BCa)", "p crudo", "p ajustado (Holm)")])),
                      html.Tbody(filas)], size="sm", striped=True, hover=True, responsive=True, className="mb-0")


def layout():
    c = contexto()
    lb = resultado("leaderboard").set_index("modelo")
    return html.Div([
        html.H1("Modelos", className="h3 mb-1"),
        html.P("Mejor combinación de cada modelo de clasificación entre las 140 evaluadas con validación anidada "
               "cronológica, y el modelo original L2E, validado con los mismos pliegues, la misma semilla y el mismo "
               "presupuesto de optimización. Las métricas externas son la media y la desviación estándar de los 5 "
               "pliegues externos; las de prueba, una sola evaluación en 2009 y 2010.", className="text-muted"),
        tarjeta("Tabla de líderes", "Ordene con los encabezados o escriba en la fila de filtros. La fila resaltada es el "
                "modelo con mayor AUC-PR externo.", tabla_lideres(),
                html.P(f"El modelo original L2E tiene el mayor AUC-PR externo ({num(lb.loc['l2e', 'auc_pr_media'], 3)}), "
                       f"frente a {num(lb.loc['naive_bayes', 'auc_pr_media'], 3)} de la mejor combinación benchmark, "
                       f"{c['mejor_global']}. Criterio de selección de cada benchmark: mayor AUC-PR medio del bucle externo "
                       "dentro de cada modelo.", className="insight mt-3 mb-0")),
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Div([dbc.Label("Modelo seleccionado", html_for="mod-selector", className="h6 mb-1 d-block"),
                              html.P("Controla la predicción, el diagnóstico y la interpretabilidad de abajo.",
                                     className="small text-muted mb-0")]), md=4),
            dbc.Col(dcc.Dropdown(id="mod-selector", options=opciones_modelo(), value=opciones_modelo()[0]["value"],
                                 clearable=False, labels=plots.ETIQUETAS_DROPDOWN), md=8),
        ], className="align-items-center")), className="seccion selector-fijo mb-4"),
        tarjeta("Predicción en vivo", "Complete los datos del cliente o cargue uno de la prueba al azar. El modelo usa solo "
                "información disponible antes de la llamada.", formulario(),
                html.Div([dbc.Button([html.I(className="bi bi-play-fill me-1"), "Predecir"], id="mod-predecir",
                                     color="primary", className="me-2"),
                          dbc.Button([html.I(className="bi bi-shuffle me-1"), "Cargar un cliente de prueba al azar"],
                                     id="mod-azar", color="secondary", outline=True)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Loading(html.Div(id="mod-resultado", **{"aria-live": "polite"}), type="dot",
                                             color=plots.AZUL), lg=4),
                         dbc.Col(grafica("mod-shap-local", 440), lg=8)], className="g-4")),
        tarjeta("Diagnóstico del modelo seleccionado", "Evaluación en el conjunto de prueba (2009-05 a 2010-11).",
                dbc.Row([dbc.Col(grafica("mod-confusion", 330), lg=6), dbc.Col(grafica("mod-roc", 330), lg=6),
                         dbc.Col(grafica("mod-pr", 330), lg=6), dbc.Col(grafica("mod-calibracion", 330), lg=6)],
                        className="g-3"),
                html.P(id="mod-diag-txt", className="insight mt-3 mb-0")),
        tarjeta("Interpretabilidad global (SHAP)", "Contribución media absoluta de cada variable original sobre una "
                "muestra de la prueba, en puntos de probabilidad.", grafica("mod-shap-global", 420),
                html.P(id="mod-shap-txt", className="insight mb-0")),
        tarjeta("Comparación estadística", "Friedman y Nemenyi con el modelo original incluido; DeLong entre los "
                "finalistas del tercer entregable, con ajuste de Holm.",
                dbc.Row([dbc.Col([dbc.RadioItems(id="mod-cd-bloques", value="pliegues", inline=True, className="mb-2",
                                                 options=[{"label": v, "value": k} for k, v in BLOQUES_CD.items()]),
                                  grafica("mod-cd", 440)], lg=7),
                         dbc.Col([html.P(friedman_texto(), className="insight"),
                                  html.H3("DeLong entre los finalistas (prueba)", className="h6"), tabla_delong()], lg=5)],
                        className="g-4")),
        tarjeta("Modelo original: logística de estado latente en dos escalas (L2E)", "Idea, algoritmo, diferencias con la "
                "literatura, convergencia, pruebas de hipótesis y veredicto. Todo se validó con los mismos pliegues "
                "externos, la misma semilla y el mismo presupuesto que los benchmarks.", *modelo_original(),
                id_="modelo-original"),
    ])


def validar(entrada: dict):
    """Revisa tipos y rangos del formulario; devuelve los valores limpios y la lista de errores."""
    limpio, errores = {}, []
    for v in esquema()["variables"]:
        nombre, valor, etiqueta = v["variable"], entrada.get(v["variable"]), v["etiqueta"]
        if valor is None or valor == "":
            errores.append(f"{etiqueta}: falta el valor.")
            continue
        if v["tipo"] == "categorica":
            if valor not in v["valores"]:
                errores.append(f"{etiqueta}: la categoría no es válida.")
            else:
                limpio[nombre] = valor
            continue
        try:
            x = float(valor)
        except (TypeError, ValueError):
            errores.append(f"{etiqueta}: debe ser un número.")
            continue
        especiales = v.get("especiales", [])
        if x not in especiales and not v["min"] <= x <= v["max"]:
            errores.append(f"{etiqueta}: debe estar entre {num(v['min'], 0 if v['entero'] else 2)} y "
                           f"{num(v['max'], 0 if v['entero'] else 2)}" + (" o ser 999." if especiales else "."))
            continue
        if v["entero"] and not x.is_integer():
            errores.append(f"{etiqueta}: debe ser un número entero.")
            continue
        limpio[nombre] = int(x) if v["entero"] else x
    return limpio, errores


@callback(Output({"tipo": "entrada", "var": ALL}, "value"), Input("mod-azar", "n_clicks"), prevent_initial_call=True)
def cargar_azar(n_clicks):
    fila = banco().query("conjunto == 'Prueba'").sample(1, random_state=int(n_clicks)).iloc[0]
    return [a_python(fila[o["id"]["var"]]) for o in ctx.outputs_list]


@callback(Output("mod-resultado", "children"), Output("mod-shap-local", "figure"), Input("mod-predecir", "n_clicks"),
          Input("mod-selector", "value"), State({"tipo": "entrada", "var": ALL}, "value"))
def predecir_cliente(_, m, __):
    entrada = {e["id"]["var"]: e.get("value") for e in ctx.states_list[0]}
    limpio, errores = validar(entrada)
    if errores:
        return (dbc.Alert([html.B("Revise los datos del cliente:"), html.Ul([html.Li(e) for e in errores], className="mb-0")],
                          color="danger"), plots.vacia("Corrija los campos indicados para ver la explicación."))
    try:
        X = fila_entrada(limpio)
        prob, clase = float(probabilidad(m, X)[0]), int(predecir(m, X)[0])
        posicion = percentil(m, float(puntaje_crudo(m, X)[0]))
        phi, base, fx = shap_permutaciones(lambda D: probabilidad(m, D), X, fondo(), VARIABLES, n_perm=64, semilla=0)
    except Exception as e:  # noqa: BLE001 - cualquier fallo se informa sin detener la aplicación
        return dbc.Alert(f"No fue posible calcular la predicción ({type(e).__name__}).", color="danger"), plots.vacia("")
    meta = modelos()[m]["meta"]
    c = contexto()
    nota = ("Estas probabilidades no están recalibradas al régimen reciente: los modelos con balanceo las inflan y los "
            f"demás reflejan la tasa del entrenamiento ({num(c['tasa_entrenamiento'] * 100)} %), mientras que en la prueba "
            f"suscribió el {num(c['tasa_prueba'] * 100)} %. Para decidir a quién llamar primero conviene usar la posición "
            "en la lista; para estimar cuántos suscribirán, hay que recalibrar con el último mes cerrado.")
    if meta["platt"]:
        nota += " En la SVM la probabilidad se obtiene con escalado de Platt sobre su función de decisión."
    if m == "l2e":
        r = resumen("l2e_resumen")
        nota = (f"El L2E usa el nivel de conversión vigente al cierre de {r['periodo_vigente']} y predice como si el "
                "cliente se llamara el día siguiente. Como ese nivel se actualiza con las etiquetas de los días cerrados, "
                "su probabilidad sí sigue el régimen reciente: en la prueba, su probabilidad media fue del "
                f"{num(r['prob_media_prueba'] * 100)} % frente a una tasa real del {num(r['tasa_prueba'] * 100)} %.")
    lugar = ("Supera a menos del 1 % de los clientes de prueba" if posicion < 0.01 else
             f"Supera al {num(posicion * 100, 0)} % de los clientes de prueba")
    resultado_ = html.Div([
        dbc.Alert([html.I(className=f"bi {'bi-telephone-forward' if clase else 'bi-telephone-x'} me-2"),
                   html.B("Predicción: suscribe" if clase else "Predicción: no suscribe"),
                   html.Span(f" (umbral de {meta['umbral']})", className="small")],
                  color="primary" if clase else "light", className="border"),
        html.Div([html.Div("Probabilidad estimada", className="cifra-texto"), html.Div(f"{num(prob * 100)} %", className="cifra")],
                 className="stat stat-claro mb-3"),
        html.Div([html.Div("Posición en la lista de llamadas", className="cifra-texto"),
                  html.Div(lugar, className="cifra-media")], className="stat stat-claro mb-3"),
        html.P(nota, className="nota mb-0"),
    ])
    valores = [valor_legible(X.iloc[0][v]) for v in VARIABLES]
    return resultado_, plots.shap_local(phi, [ETIQUETAS[v] for v in VARIABLES], valores, base, fx)


@callback(Output("mod-confusion", "figure"), Output("mod-roc", "figure"), Output("mod-pr", "figure"),
          Output("mod-calibracion", "figure"), Output("mod-diag-txt", "children"), Output("mod-shap-global", "figure"),
          Output("mod-shap-txt", "children"), Input("mod-selector", "value"))
def diagnostico(m):
    lb = resultado("leaderboard").set_index("modelo").loc[m]
    curvas = resultado("curvas").query("modelo == @m")
    conf = resultado("confusion").set_index("modelo").loc[m]
    cal = resultado("calibracion").query("modelo == @m")
    prev = contexto()["tasa_prueba"]
    total = conf[["tn", "fp", "fn", "tp"]].sum()
    recall = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
    txt = (f"En prueba, {lb['nombre']} obtiene un AUC de {num(lb['auc_prueba'], 3)} y un AUC-PR de "
           f"{num(lb['auc_pr_prueba'], 3)}, frente a {num(prev, 3)} de un modelo al azar. Con el umbral por defecto marca "
           f"como suscriptores al {num((conf['tp'] + conf['fp']) / total * 100)} % de los clientes y encuentra al "
           f"{num(recall * 100)} % de quienes suscriben. El ECE de {num(lb['ece_prueba'], 3)} mide cuánto se apartan sus "
           "probabilidades de la frecuencia real: " + ("es bajo porque el nivel diario del modelo sigue el cambio de "
                                                       "prevalencia entre 2008 y 2009." if m == "l2e" else
                                                       "el cambio de prevalencia entre 2008 y 2009 las descalibra."))
    shap = resultado("shap_global").query("modelo == @m")
    imp = shap.groupby("variable")["phi"].agg(lambda s: s.abs().mean()).sort_values(ascending=False)
    txt_shap = (f"Las variables con mayor contribución media son {ETIQUETAS[imp.index[0]].lower()}, "
                f"{ETIQUETAS[imp.index[1]].lower()} y {ETIQUETAS[imp.index[2]].lower()}. Son asociaciones aprendidas en el "
                "periodo de entrenamiento, no efectos causales; si el contexto económico cambia, su peso también cambia.")
    return (plots.confusion(conf), plots.roc(curvas.query("tipo == 'roc'"), lb["auc_prueba"]),
            plots.pr(curvas.query("tipo == 'pr'"), lb["auc_pr_prueba"], prev),
            plots.calibracion(cal, lb["brier_prueba"], lb["ece_prueba"]), txt,
            plots.shap_global(shap, ETIQUETAS), txt_shap)


@callback(Output("mod-cd", "figure"), Input("mod-cd-bloques", "value"))
def diagrama_cd(bloques):
    rangos = resultado("cd").query("bloques == @bloques")
    return plots.diferencia_critica(rangos, destacado="L2E (modelo original)", subtitulo=f"Bloques: {BLOQUES_CD[bloques]}")
