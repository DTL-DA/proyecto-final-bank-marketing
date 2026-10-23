"""Sección 3 · Modelos: tabla de líderes, predicción en vivo con SHAP local, diagnóstico, SHAP global y comparación."""
import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from src import plots
from src.data import (banco, contexto, esquema, fila_entrada, fondo, modelos, opciones_modelo, percentil, predecir,
                      probabilidad, puntaje_crudo, resultado)
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


def friedman_texto():
    fr = resultado("friedman")
    glob, pl, me = fr.iloc[0], fr.iloc[1], fr.iloc[2]
    return (f"Sobre las {int(glob['Modelos (k)'])} combinaciones de clasificación, Friedman da χ² = {num(glob['χ²_F'])} "
            f"(p {glob['p (χ²)']}): al menos una difiere, sobre todo por las peores. Entre los 7 finalistas no hay "
            f"evidencia de diferencias: p = {pl['p (χ²)']} con los pliegues como bloques y p = {me['p (χ²)']} con bloques "
            f"mensuales, y todos quedan dentro de la diferencia crítica de {num(pl['Diferencia crítica de Nemenyi'], 2)} "
            "rangos. Por eso la mejor combinación global pertenece al conjunto de finalistas, pero no se puede declarar "
            "un ganador.")


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
    return html.Div([
        html.H1("Modelos", className="h3 mb-1"),
        html.P("Mejor combinación de cada modelo de clasificación entre las 140 evaluadas con validación anidada "
               "cronológica. Las métricas externas son la media y la desviación estándar de los 5 pliegues externos; las "
               "de prueba, una sola evaluación en 2009 y 2010.", className="text-muted"),
        tarjeta("Tabla de líderes", "Ordene con los encabezados o escriba en la fila de filtros. La fila resaltada es la "
                "mejor combinación global por AUC-PR externo.", tabla_lideres(),
                html.P(f"Mejor combinación global: {c['mejor_global']}. Criterio de selección: mayor AUC-PR medio del "
                       "bucle externo dentro de cada modelo.", className="insight mt-3 mb-0")),
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
        tarjeta("Comparación estadística", "Friedman y Nemenyi sobre los pliegues externos; DeLong sobre la prueba con "
                "ajuste de Holm.",
                dbc.Row([dbc.Col(dcc.Graph(figure=plots.diferencia_critica(resultado("cd").query("bloques == 'pliegues'")),
                                           config=plots.CONFIG, responsive=True, style={"height": "420px"}), lg=7),
                         dbc.Col([html.P(friedman_texto(), className="insight"), tabla_delong()], lg=5)], className="g-4")),
        tarjeta("Modelo original", None,
                dbc.Alert([html.I(className="bi bi-hourglass-split me-2"), html.B("En desarrollo. "),
                           "Esta subsección mostrará la descripción y el pseudocódigo del modelo propio, sus diferencias "
                           "frente a la literatura, la curva de convergencia, la tabla de pruebas de hipótesis con p-valores "
                           "ajustados, el tamaño del efecto y el veredicto de superioridad. El modelo se valida con los "
                           "mismos pliegues externos y la misma semilla que los benchmarks, y entra al diagrama de diferencia "
                           "crítica como un competidor más."], color="info", className="mb-0"), id_="modelo-original"),
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
           "probabilidades de la frecuencia real: el cambio de prevalencia entre 2008 y 2009 las descalibra.")
    shap = resultado("shap_global").query("modelo == @m")
    imp = shap.groupby("variable")["phi"].agg(lambda s: s.abs().mean()).sort_values(ascending=False)
    txt_shap = (f"Las variables con mayor contribución media son {ETIQUETAS[imp.index[0]].lower()}, "
                f"{ETIQUETAS[imp.index[1]].lower()} y {ETIQUETAS[imp.index[2]].lower()}. Son asociaciones aprendidas en el "
                "periodo de entrenamiento, no efectos causales; si el contexto económico cambia, su peso también cambia.")
    return (plots.confusion(conf), plots.roc(curvas.query("tipo == 'roc'"), lb["auc_prueba"]),
            plots.pr(curvas.query("tipo == 'pr'"), lb["auc_pr_prueba"], prev),
            plots.calibracion(cal, lb["brier_prueba"], lb["ece_prueba"]), txt,
            plots.shap_global(shap, ETIQUETAS), txt_shap)
