"""Sección 2 · EDA: filtros globales, distribuciones, explorador bivariado, correlación, faltantes y atípicos."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import banco

dash.register_page(__name__, path="/eda", name="EDA", order=1, title="EDA · Bank Marketing")

NUMERICAS = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
             "euribor3m", "nr.employed"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
ETIQUETAS = {"age": "Edad (años)", "duration": "Duración de la llamada (s)", "campaign": "Contactos en la campaña",
             "pdays": "Días desde el último contacto", "previous": "Contactos previos",
             "emp.var.rate": "Variación del empleo", "cons.price.idx": "Índice de precios",
             "cons.conf.idx": "Índice de confianza", "euribor3m": "Euribor a 3 meses (%)",
             "nr.employed": "Número de empleados", "job": "Ocupación", "marital": "Estado civil",
             "education": "Nivel educativo", "default": "Crédito en mora", "housing": "Crédito de vivienda",
             "loan": "Crédito personal", "contact": "Canal de contacto", "month": "Mes", "day_of_week": "Día de la semana",
             "poutcome": "Resultado de la campaña anterior", "suscribe": "Suscripción (tasa)", "conjunto": "Conjunto"}
COLORES = ["conjunto", "contact", "default", "housing", "loan", "poutcome", "marital", "day_of_week"]


def num(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def opciones(variables):
    return [{"label": ETIQUETAS[v], "value": v} for v in variables]


def grafica(id_grafica, id_texto, altura, controles=None):
    # Altura explícita igual a la de la figura: la tarjeta reserva el espacio antes de que Plotly dibuje
    cuerpo = ([controles] if controles else []) + [
        dcc.Loading(dcc.Graph(id=id_grafica, config=plots.CONFIG, responsive=True, style={"height": f"{altura}px"}),
                    type="dot", color=plots.AZUL),
        html.P(id=id_texto, className="insight mb-0")]
    return dbc.Card(dbc.CardBody(cuerpo), className="seccion h-100")


def desplegable(id_, opciones_, valor, **kwargs):
    return dcc.Dropdown(id=id_, options=opciones_, value=valor, labels=plots.ETIQUETAS_DROPDOWN, **kwargs)


def layout():
    df = banco()
    inicio, fin = df["fecha"].min(), df["fecha"].max() + pd.offsets.MonthEnd(0)
    filtros = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([dbc.Label("Periodo de contacto", html_for="eda-fechas"),
                 dcc.DatePickerRange(id="eda-fechas", min_date_allowed=inicio, max_date_allowed=fin, start_date=inicio,
                                     end_date=fin, display_format="MM/YYYY", first_day_of_week=1, className="d-block")],
                md=4),
        dbc.Col([dbc.Label("Edad del cliente", html_for="eda-edad"),
                 dcc.RangeSlider(id="eda-edad", min=int(df["age"].min()), max=int(df["age"].max()), step=1,
                                 value=[int(df["age"].min()), int(df["age"].max())], allowCross=False,
                                 marks={a: str(a) for a in range(20, 100, 10)},
                                 tooltip={"placement": "bottom", "always_visible": False})], md=3),
        dbc.Col([dbc.Label("Conjunto", html_for="eda-conjunto"),
                 desplegable("eda-conjunto", ["Todos", "Entrenamiento", "Prueba"], "Todos", clearable=False)], md=2),
        dbc.Col([dbc.Label("Canal de contacto", html_for="eda-canal"),
                 desplegable("eda-canal", [{"label": "Celular", "value": "cellular"},
                                           {"label": "Teléfono fijo", "value": "telephone"}],
                             ["cellular", "telephone"], multi=True, placeholder="Todos los canales")], md=3),
    ], className="g-3 align-items-end")), className="seccion filtros mb-2")

    return html.Div([
        html.H1("Análisis exploratorio", className="h3 mb-1"),
        html.P("Los filtros actualizan todas las gráficas de la página. Cada gráfica incluye una lectura calculada con los "
               "datos seleccionados.", className="text-muted"),
        dbc.Container(filtros, fluid=True, className="px-0 filtros-fijos"),
        html.P(id="eda-resumen", className="small text-muted mb-3", **{"aria-live": "polite"}),
        dbc.Row([dbc.Col(grafica("eda-objetivo", "eda-objetivo-txt", 340), lg=5),
                 dbc.Col(grafica("eda-mensual", "eda-mensual-txt", 340), lg=7)], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(grafica("eda-uni", "eda-uni-txt", 420, dbc.Row(dbc.Col([
                dbc.Label("Variable", html_for="eda-uni-var"),
                desplegable("eda-uni-var", opciones(NUMERICAS + CATEGORICAS), "age", clearable=False)], md=8))), lg=6),
            dbc.Col(grafica("eda-bi", "eda-bi-txt", 420, dbc.Row([
                dbc.Col([dbc.Label("Eje x", html_for="eda-bi-x"),
                         desplegable("eda-bi-x", opciones(NUMERICAS + CATEGORICAS), "poutcome", clearable=False)], md=4),
                dbc.Col([dbc.Label("Eje y", html_for="eda-bi-y"),
                         desplegable("eda-bi-y", opciones(["suscribe"] + NUMERICAS + CATEGORICAS), "suscribe",
                                     clearable=False)], md=4),
                dbc.Col([dbc.Label("Color", html_for="eda-bi-color"),
                         desplegable("eda-bi-color", opciones(COLORES), None, placeholder="Sin color")], md=4)],
                className="g-2")), lg=6),
        ], className="g-4 mb-4"),
        dbc.Row(dbc.Col(grafica("eda-corr", "eda-corr-txt", 520)), className="mb-4"),
        dbc.Row([dbc.Col(grafica("eda-falt", "eda-falt-txt", 380), lg=6),
                 dbc.Col(grafica("eda-atip", "eda-atip-txt", 380), lg=6)], className="g-4"),
    ])


def filtrar(inicio, fin, edad, conjunto, canales) -> pd.DataFrame:
    df = banco()
    sel = df["fecha"].between(pd.Timestamp(inicio).normalize(), pd.Timestamp(fin)) & df["age"].between(*edad)
    if conjunto != "Todos":
        sel &= df["conjunto"] == conjunto
    if canales:
        sel &= df["contact"].isin(canales)
    return df[sel]


FILTROS = [Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"), Input("eda-edad", "value"),
           Input("eda-conjunto", "value"), Input("eda-canal", "value")]
SIN_DATOS = "No hay registros con estos filtros; amplíe el periodo o la edad."


@callback(Output("eda-resumen", "children"), Output("eda-objetivo", "figure"), Output("eda-objetivo-txt", "children"),
          Output("eda-mensual", "figure"), Output("eda-mensual-txt", "children"), Output("eda-corr", "figure"),
          Output("eda-corr-txt", "children"), Output("eda-falt", "figure"), Output("eda-falt-txt", "children"),
          Output("eda-atip", "figure"), Output("eda-atip-txt", "children"), *FILTROS)
def actualizar_generales(inicio, fin, edad, conjunto, canales):
    df = filtrar(inicio, fin, edad, conjunto, canales)
    if df.empty:
        v = plots.vacia(SIN_DATOS)
        return "0 registros seleccionados.", v, SIN_DATOS, v, "", v, "", v, "", v, ""
    total = len(banco())
    resumen = f"{num(len(df), 0)} de {num(total, 0)} registros seleccionados ({num(len(df) / total * 100)} %)."

    tasa = df["suscribe"].mean()
    txt_obj = (f"Suscribe el {num(tasa * 100)} % de los clientes seleccionados: hay {num((1 - tasa) / max(tasa, 1e-9))} "
               "que no suscriben por cada uno que sí. Con este desbalance, el accuracy no sirve como métrica principal.")

    g = df.groupby("periodo").agg(tasa=("suscribe", "mean"), n=("suscribe", "size"))
    txt_mes = (f"La tasa mensual va del {num(g['tasa'].min() * 100)} % ({g['tasa'].idxmin()}) al "
               f"{num(g['tasa'].max() * 100)} % ({g['tasa'].idxmax()}).")
    if len(g) >= 4:
        rho = g["tasa"].corr(g["n"], method="spearman")
        txt_mes += (f" Los meses con más contactos tienden a tener tasas más bajas (Spearman entre volumen y tasa = "
                    f"{num(rho, 2)})." if rho < 0 else f" La correlación de Spearman entre volumen y tasa es {num(rho, 2)}.")

    c = df[NUMERICAS].corr(method="spearman").where(~np.eye(len(NUMERICAS), dtype=bool))
    a, b = np.unravel_index(np.nanargmax(np.abs(c.to_numpy())), c.shape)
    txt_corr = (f"El par más correlacionado es {ETIQUETAS[NUMERICAS[a]].lower()} con {ETIQUETAS[NUMERICAS[b]].lower()} "
                f"(ρ = {num(c.iloc[a, b], 2)}): los indicadores macroeconómicos se mueven juntos porque marcan el mismo "
                "periodo, por eso el modelo retira dos de ellos por VIF.")

    unk = {v: (df[v] == "unknown").mean() for v in CATEGORICAS}
    peor = max(unk, key=unk.get)
    txt_falt = (f"La variable con más valores unknown es {ETIQUETAS[peor].lower()} ({num(unk[peor] * 100)} %), y "
                f"pdays = 999 aparece en el {num((df['pdays'] == 999).mean() * 100)} % de los registros. Unknown se trata "
                "como una categoría porque es informativo.")

    atip = {}
    for v in NUMERICAS:
        s = df[v] if v != "pdays" else df.loc[df[v] != 999, v]
        if len(s) >= 5:
            q1, q3 = s.quantile([0.25, 0.75])
            atip[v] = ((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).mean()
    mayor = max(atip, key=atip.get)
    txt_atip = (f"La variable con más valores fuera de las cercas de Tukey es {ETIQUETAS[mayor].lower()} "
                f"({num(atip[mayor] * 100)} %). Son valores plausibles de colas largas, no errores, así que se conservan.")

    return (resumen, plots.objetivo(df), txt_obj, plots.tasa_mensual(df), txt_mes,
            plots.correlacion(df, NUMERICAS, ETIQUETAS), txt_corr, plots.faltantes(df, CATEGORICAS, ETIQUETAS), txt_falt,
            plots.atipicos(df, NUMERICAS, ETIQUETAS), txt_atip)


@callback(Output("eda-uni", "figure"), Output("eda-uni-txt", "children"), Input("eda-uni-var", "value"), *FILTROS)
def actualizar_univariada(var, inicio, fin, edad, conjunto, canales):
    df = filtrar(inicio, fin, edad, conjunto, canales)
    if df.empty:
        return plots.vacia(SIN_DATOS), SIN_DATOS
    numerica = var in NUMERICAS
    fig = plots.univariada(df, var, ETIQUETAS[var], numerica)
    if numerica:
        s = df[var] if var != "pdays" else df.loc[df[var] != 999, var]
        if s.empty:
            return fig, "Ningún registro seleccionado tiene contacto previo registrado."
        d = 0 if pd.api.types.is_integer_dtype(s) else 2
        txt = (f"Mediana de {num(s.median(), d)}, entre {num(s.min(), d)} y {num(s.max(), d)}, con asimetría de "
               f"{num(s.skew(), 2)}" + (": una cola larga a la derecha que en el modelo se suaviza con log1p."
                                      if s.skew() > 1 else "."))
    else:
        f = df[var].value_counts(normalize=True)
        txt = (f"La categoría más frecuente es {f.index[0]} ({num(f.iloc[0] * 100)} %); la menos frecuente, {f.index[-1]} "
               f"({num(f.iloc[-1] * 100)} %).")
    return fig, txt


@callback(Output("eda-bi", "figure"), Output("eda-bi-txt", "children"), Input("eda-bi-x", "value"),
          Input("eda-bi-y", "value"), Input("eda-bi-color", "value"), *FILTROS)
def actualizar_bivariada(x, y, color, inicio, fin, edad, conjunto, canales):
    df = filtrar(inicio, fin, edad, conjunto, canales)
    if df.empty:
        return plots.vacia(SIN_DATOS), SIN_DATOS
    if x == y:
        return plots.vacia("Elija variables distintas para los dos ejes."), ""
    numericas = set(NUMERICAS)
    fig = plots.bivariada(df, x, y, color, ETIQUETAS, numericas)
    if y == "suscribe" and x not in numericas:
        g = df.groupby(x)["suscribe"].agg(["mean", "size"]).query("size >= 30")
        if g.empty:
            return fig, "Hay muy pocos registros por categoría para comparar tasas."
        txt = (f"La tasa de suscripción va del {num(g['mean'].min() * 100)} % ({g['mean'].idxmin()}) al "
               f"{num(g['mean'].max() * 100)} % ({g['mean'].idxmax()}), contando solo categorías con 30 registros o más.")
    elif y == "suscribe":
        rho = df[x].corr(df["suscribe"], method="spearman")
        txt = (f"La correlación de Spearman entre {ETIQUETAS[x].lower()} y la suscripción es {num(rho, 2)}; la curva "
               "muestra si la relación es monótona o tiene forma de U.")
    elif x in numericas and y in numericas:
        txt = (f"La correlación de Spearman entre {ETIQUETAS[x].lower()} y {ETIQUETAS[y].lower()} es "
               f"{num(df[x].corr(df[y], method='spearman'), 2)}.")
    elif x in numericas or y in numericas:
        cat, nm = (x, y) if x not in numericas else (y, x)
        med = df.groupby(cat)[nm].median()
        txt = (f"La mediana de {ETIQUETAS[nm].lower()} va de {num(med.min(), 2)} ({med.idxmin()}) a {num(med.max(), 2)} "
               f"({med.idxmax()}).")
    else:
        tabla = pd.crosstab(df[x], df[y], normalize="index")
        fila, col = np.unravel_index(np.argmax(tabla.to_numpy()), tabla.shape)
        txt = (f"Entre los clientes con {ETIQUETAS[x].lower()} = {tabla.index[fila]}, el "
               f"{num(tabla.iloc[fila, col] * 100)} % tiene {ETIQUETAS[y].lower()} = {tabla.columns[col]}: es la mayor "
               "concentración de la tabla.")
    return fig, txt
