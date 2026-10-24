"""Punto de entrada del dashboard predictivo de Bank Marketing (proyecto final, cuarto entregable: modelo original).

Local:   python app.py        →  http://127.0.0.1:8050
Render:  gunicorn app:server
"""
import os

# Un solo hilo de BLAS y OpenMP: con más hilos KNN resuelve distinto los empates de distancia y sus predicciones dejan
# de coincidir con las del cuaderno; además evita competir por la CPU del plan gratuito.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import html  # noqa: E402

from src import plots  # noqa: E402,F401  registra la plantilla común de Plotly
from src.data import banco, contexto, modelos  # noqa: E402

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
                title="Bank Marketing · Proyecto final", suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
                           {"name": "description", "content": "Dashboard predictivo de suscripción a depósitos a término"}])
server = app.server
JBOOK = "https://dtl-da.github.io/proyecto-final-bank-marketing/cuarto-entregable/"

# Datos y modelos se cargan una sola vez, al iniciar el servidor
banco()
modelos()

navbar = dbc.NavbarSimple(
    children=[dbc.NavItem(dbc.NavLink(p["name"], href=p["relative_path"], active="exact"))
              for p in dash.page_registry.values()],
    brand="Bank Marketing · Proyecto final", brand_href="/", color="primary", dark=True, sticky="top")

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid="xl", className="py-4"),
    html.Footer(dbc.Container(html.Small([
        f"Donny Torres · Maestría en Analítica de Datos, Universidad del Norte · Versión {contexto()['version']} · "
        "Datos: UCI Machine Learning Repository (CC BY 4.0) · ",
        html.A("Jupyter Book del proyecto", href=JBOOK, target="_blank",
               rel="noopener noreferrer", className="enlace-pie")]), fluid="xl"), className="pie"),
])

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
