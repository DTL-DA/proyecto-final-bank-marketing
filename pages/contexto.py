"""Sección 1 · Contexto: problema, pregunta de investigación, datos, metodología, limitaciones y autoría."""
from datetime import date

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src import plots
from src.data import banco, contexto

dash.register_page(__name__, path="/", name="Contexto", order=0, title="Contexto · Bank Marketing")

DICCIONARIO = [
    ("age", "Cliente", "Edad del cliente (años)", "Predictora"),
    ("job", "Cliente", "Tipo de ocupación (12 categorías, incluye unknown)", "Predictora"),
    ("marital", "Cliente", "Estado civil", "Predictora"),
    ("education", "Cliente", "Nivel educativo (8 niveles, incluye unknown)", "Predictora"),
    ("default", "Cliente", "¿Tiene crédito en mora?", "Predictora"),
    ("housing", "Cliente", "¿Tiene crédito de vivienda?", "Predictora"),
    ("loan", "Cliente", "¿Tiene crédito personal?", "Predictora"),
    ("contact", "Último contacto", "Canal del contacto (celular o teléfono fijo)", "Predictora"),
    ("month", "Último contacto", "Mes del último contacto (sin año)", "Predictora"),
    ("day_of_week", "Último contacto", "Día de la semana del último contacto", "Predictora"),
    ("duration", "Último contacto", "Duración de la llamada (s); solo se conoce al terminarla", "Excluida por fuga"),
    ("campaign", "Campaña actual", "Contactos a este cliente en la campaña, incluido el último", "Predictora"),
    ("pdays", "Campañas anteriores", "Días desde el último contacto previo (999 = sin contacto)", "Predictora"),
    ("previous", "Campañas anteriores", "Contactos antes de esta campaña", "Predictora"),
    ("poutcome", "Campañas anteriores", "Resultado de la campaña anterior", "Predictora"),
    ("emp.var.rate", "Contexto económico", "Tasa de variación del empleo (trimestral)", "Excluida por VIF"),
    ("cons.price.idx", "Contexto económico", "Índice de precios al consumidor (mensual)", "Predictora"),
    ("cons.conf.idx", "Contexto económico", "Índice de confianza del consumidor (mensual)", "Predictora"),
    ("euribor3m", "Contexto económico", "Tasa Euribor a tres meses (%)", "Excluida por VIF"),
    ("nr.employed", "Contexto económico", "Número de empleados (trimestral)", "Predictora"),
    ("y", "Objetivo", "¿El cliente suscribió un depósito a término?", "Variable objetivo"),
]
PASOS = [("bi-database", "ETL", "Limpieza, códigos especiales y auditoría de fuga"),
         ("bi-bar-chart", "EDA", "Distribuciones, deriva y cambio de régimen"),
         ("bi-diagram-3", "7 modelos", "KNN, Naive Bayes, logística, árbol, bosque, XGBoost y SVM"),
         ("bi-sliders", "4 balanceos", "Sin balanceo, SMOTE, ADASYN y class_weight"),
         ("bi-search", "4 optimizadores", "Grid, Random, Bayesiana y Genético"),
         ("bi-layers", "Validación anidada", "5 pliegues externos y 3 internos, cronológicos"),
         ("bi-graph-up-arrow", "Comparación estadística", "Friedman, Nemenyi, DeLong y tamaño del efecto"),
         ("bi-lightbulb", "Modelo original", "Propuesta propia validada con las mismas reglas")]


def num(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def tarjeta(titulo, *hijos, clase=""):
    return dbc.Card(dbc.CardBody([html.H2(titulo, className="h5 card-title"), *hijos]), className=f"seccion h-100 {clase}")


def layout():
    c = contexto()
    cifras = [(num(c["registros"], 0), "llamadas registradas"), (f"{num(c['tasa_total'] * 100)} %", "de los clientes suscribe"),
              (str(c["periodos"]), f"meses, de {c['periodo_inicio']} a {c['periodo_fin']}"),
              (str(c["combinaciones"]), "combinaciones evaluadas")]
    hero = html.Section([
        html.P("Proyecto final · Tercer entregable · Machine Learning", className="eyebrow"),
        html.H1("¿A quién llamar primero?", className="display-5 fw-bold"),
        html.P("Un banco portugués vende depósitos a término por teléfono. Este dashboard muestra qué tanto se puede "
               "anticipar, con la información disponible antes de la llamada, si un cliente suscribirá, y permite usar "
               "los mejores modelos para predecir y explicar cada predicción.", className="lead"),
        dbc.Row([dbc.Col(html.Div([html.Div(v, className="cifra"), html.Div(t, className="cifra-texto")], className="stat"),
                         xs=6, lg=3) for v, t in cifras], className="g-3 mt-2"),
    ], className="hero mb-4")

    importa = tarjeta(
        "Por qué importa",
        html.P("Cada llamada cuesta tiempo de un asesor y molesta a quien no está interesado. Ordenar bien la lista de "
               "llamadas permite encontrar a más suscriptores con menos contactos."),
        html.P(["Los datos provienen de las campañas de telemercadeo de un banco portugués entre mayo de 2008 y "
                "noviembre de 2010, en plena crisis financiera (", html.A("Moro, Cortez y Rita, 2014",
                href="https://doi.org/10.1016/j.dss.2014.03.001", target="_blank", rel="noopener noreferrer"),
                "). En ese periodo la Euribor a tres meses pasó de ", html.B(f"{num(c['euribor_inicio'], 2)} %"),
                " en mayo de 2008 a ", html.B(f"{num(c['euribor_2009_05'], 2)} %"), " en mayo de 2009, y la tasa de "
                "suscripción pasó del ", html.B(f"{num(c['tasa_entrenamiento'] * 100)} %"), " en el entrenamiento al ",
                html.B(f"{num(c['tasa_prueba'] * 100)} %"), " en la prueba."]),
        html.P("Fuente de las cifras: cálculo propio con las variables y y euribor3m del conjunto de datos.",
               className="nota"),
        dcc.Graph(figure=plots.tasa_mensual(banco()), config=plots.CONFIG, responsive=True, style={"height": "340px"}),
        html.P("La tasa de suscripción no es estable: cambia de régimen entre 2008 y 2009, y eso condiciona la "
               "validación y la lectura de las probabilidades.", className="insight"))

    pregunta = tarjeta(
        "Pregunta de investigación y objetivos",
        html.P(html.B("¿Qué tan bien se puede anticipar, con información previa a la llamada, si un cliente suscribirá "
                      "un depósito a término, y qué combinación de modelo, balanceo y optimizador lo hace mejor?")),
        html.Ol([html.Li("Comparar 7 modelos, 4 técnicas de balanceo y 5 optimizadores de hiperparámetros con validación "
                         "anidada cronológica."),
                 html.Li("Elegir la mejor combinación de cada modelo con criterios estadísticos, no visuales."),
                 html.Li("Poner los modelos a disposición para predecir y explicar cada predicción con SHAP."),
                 html.Li("Proponer un modelo original y verificar si supera a los benchmarks con pruebas de hipótesis.")]),
        html.Hr(),
        html.H3("Los datos", className="h6"),
        html.Ul([html.Li(["Fuente: ", html.A("UCI Machine Learning Repository", href="https://archive.ics.uci.edu/dataset/222/bank+marketing",
                                             target="_blank", rel="noopener noreferrer"), " (licencia CC BY 4.0)."]),
                 html.Li(f"Periodo: {c['periodo_inicio']} a {c['periodo_fin']} ({c['periodos']} meses). El archivo no trae "
                         "el año; se reconstruye por el orden de las filas."),
                 html.Li(f"Observaciones: {num(c['registros'], 0)} ({num(c['registros_sin_duplicados'], 0)} sin duplicados "
                         "exactos) y 20 variables predictoras más la objetivo y."),
                 html.Li("Tarea: clasificación binaria (suscribe o no) con información previa a la llamada."),
                 html.Li(f"Partición cronológica 80/20: entrenamiento de {num(c['entrenamiento'], 0)} registros y prueba de "
                         f"{num(c['prueba'], 0)}, usada una sola vez.")]))

    tabla = dbc.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Grupo"), html.Th("Significado"), html.Th("Uso")])),
                       html.Tbody([html.Tr([html.Td(html.Code(v)), html.Td(g), html.Td(s), html.Td(u)])
                                   for v, g, s, u in DICCIONARIO])],
                      bordered=False, hover=True, striped=True, size="sm", responsive=True, className="mb-0")

    pasos = []
    for i, (icono, titulo, texto) in enumerate(PASOS):
        pendiente = titulo == "Modelo original"
        pasos.append(html.Div([html.I(className=f"bi {icono} paso-icono"), html.Div(titulo, className="paso-titulo"),
                               html.Div(texto + (" (en desarrollo)" if pendiente else ""), className="paso-texto")],
                              className="paso paso-pendiente" if pendiente else "paso", role="listitem"))
        if i < len(PASOS) - 1:
            pasos.append(html.I(className="bi bi-arrow-right paso-flecha", **{"aria-hidden": "true"}))
    metodologia = tarjeta(
        "Metodología",
        html.Div(pasos, className="pipeline", role="list", lang="es", **{"aria-label": "Etapas del pipeline del proyecto"}),
        html.P(f"Cada combinación se evaluó en 5 pliegues externos con 20 configuraciones por optimizador: "
               f"{num(c['evaluaciones'], 0)} evaluaciones internas y {num(c['horas_cpu'])} horas de CPU. Las métricas de la "
               "página de modelos vienen del bucle externo, nunca del valor óptimo del bucle interno.", className="insight mt-3"))

    limites = tarjeta(
        "Limitaciones, ética y sesgo",
        html.Ul([
            html.Li([html.B("Cambio de régimen. "), "Los modelos aprenden sobre todo de 2008 y se aplican a 2009 y 2010, con "
                     "otra tasa de suscripción; sus probabilidades reflejan la tasa del entrenamiento."]),
            html.Li([html.B("Población. "), "Solo clientes que el banco eligió contactar, de una institución y un país; los "
                     "resultados no se generalizan a otros bancos o productos sin validación."]),
            html.Li([html.B("Información sensible y equidad. "), "default, housing y loan describen la situación financiera; "
                     "priorizar llamadas por edad u ocupación puede concentrar la presión comercial en grupos específicos, "
                     "por lo que un uso real debe auditar el efecto por grupos."]),
            html.Li([html.B("Privacidad. "), "El archivo no tiene nombres, documentos ni identificadores, y las fechas no tienen "
                     "año ni día; el riesgo de reidentificación es bajo."]),
        ]),
        html.Hr(),
        html.P([html.B("Autor: "), "Donny Torres · Maestría en Analítica de Datos, Universidad del Norte"], className="mb-1"),
        html.P([html.B("Versión: "), f"{c['version']} · actualizada el {date.fromisoformat(c['actualizado']):%d/%m/%Y}"],
               className="mb-1"),
        html.P([html.B("Datos: "), "Moro, S., Cortez, P. y Rita, P. (2014). A data-driven approach to predict the success "
                "of bank telemarketing. Decision Support Systems, 62, 22–31."], className="nota mb-0"))

    return html.Div([
        hero,
        dbc.Row([dbc.Col(importa, lg=7), dbc.Col(pregunta, lg=5)], className="g-4 mb-4"),
        dbc.Row(dbc.Col(metodologia), className="mb-4"),
        dbc.Row([dbc.Col(tarjeta("Diccionario de variables", tabla), lg=7), dbc.Col(limites, lg=5)], className="g-4"),
    ])
