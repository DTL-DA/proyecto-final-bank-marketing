"""Funciones que construyen las figuras del dashboard, con una plantilla de Plotly común."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

TINTA, TINTA_2, GRIS, REJILLA = "#2C3E50", "#4A5568", "#8A94A6", "#E6E8EC"
AZUL, NARANJA = "#2A78D6", "#EB6834"
PALETA = [AZUL, NARANJA, "#1BAF7A", "#EDA100", "#E87BA4"]
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}
ETIQUETAS_DROPDOWN = {"select_all": "Seleccionar todo", "deselect_all": "Quitar todo",
                      "selected_count": "{num_selected} seleccionados", "search": "Buscar",
                      "clear_search": "Borrar búsqueda", "clear_selection": "Quitar selección",
                      "no_options_found": "Sin opciones"}

pio.templates["bm"] = go.layout.Template(layout=dict(
    font=dict(family="Lato, 'Segoe UI', Arial, sans-serif", size=13, color=TINTA),
    title=dict(font=dict(size=15, color=TINTA), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=PALETA,
    xaxis=dict(gridcolor=REJILLA, linecolor="#C9CED6", zerolinecolor=REJILLA, ticks="", automargin=True),
    yaxis=dict(gridcolor=REJILLA, linecolor="#C9CED6", zerolinecolor=REJILLA, ticks="", automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    hoverlabel=dict(font_size=13, font_family="Lato, 'Segoe UI', Arial, sans-serif"),
    margin=dict(l=10, r=10, t=56, b=10), separators=", ",
))
pio.templates.default = "bm"


def _num(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def vacia(mensaje: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_2), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(height=320)


# ------------------------------------------------------------------------------------------------------------- EDA
def objetivo(df: pd.DataFrame) -> go.Figure:
    conteo = df["suscribe"].value_counts().reindex([0, 1], fill_value=0)
    pct = conteo / max(conteo.sum(), 1) * 100
    fig = go.Figure(go.Bar(x=["No suscribe", "Suscribe"], y=conteo.values, marker_color=[GRIS, AZUL],
                           text=[f"{_num(p)} %" for p in pct], textposition="outside", width=0.5, cliponaxis=False,
                           hovertemplate="%{x}: %{y:,} clientes<extra></extra>"))
    fig.update_yaxes(title_text="Clientes", range=[0, max(conteo.max(), 1) * 1.15], tickformat=",d")
    return fig.update_layout(title="Distribución de la variable objetivo", height=340, showlegend=False)


def tasa_mensual(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("periodo").agg(tasa=("suscribe", "mean"), n=("suscribe", "size")).reset_index()
    fig = go.Figure(go.Scatter(x=g["periodo"], y=g["tasa"] * 100, mode="lines+markers", line=dict(color=AZUL, width=2),
                               marker=dict(size=7, color=AZUL, line=dict(color="white", width=1.5)), customdata=g["n"],
                               hovertemplate="%{x}: %{y:.1f} % suscribe<br>%{customdata:,} contactos<extra></extra>"))
    fig.update_yaxes(title_text="Clientes que suscriben (%)", rangemode="tozero")
    fig.update_xaxes(title_text="Mes de contacto", type="category")
    return fig.update_layout(title="Tasa de suscripción por mes", height=340, showlegend=False)


def univariada(df: pd.DataFrame, var: str, etiqueta: str, numerica: bool) -> go.Figure:
    if numerica:
        serie = df[var] if var != "pdays" else df.loc[df[var] != 999, var]
        fig = go.Figure(go.Histogram(x=serie, nbinsx=40, marker_color=AZUL, marker_line=dict(color="white", width=1),
                                     hovertemplate=f"{etiqueta}: %{{x}}<br>%{{y:,}} clientes<extra></extra>"))
        fig.update_xaxes(title_text=etiqueta + (" (sin el código 999)" if var == "pdays" else ""))
    else:
        conteo = df[var].value_counts()
        fig = go.Figure(go.Bar(x=conteo.index, y=conteo.values, marker_color=AZUL,
                               customdata=conteo.values / max(conteo.sum(), 1) * 100,
                               hovertemplate="%{x}: %{y:,} clientes (%{customdata:.1f} %)<extra></extra>"))
        fig.update_xaxes(title_text=etiqueta, type="category")
    fig.update_yaxes(title_text="Clientes", tickformat=",d")
    return fig.update_layout(title=f"Distribución de {etiqueta.lower()}", height=420, showlegend=False, bargap=0.15)


def bivariada(df: pd.DataFrame, x: str, y: str, color: str | None, etiquetas: dict, numericas: set) -> go.Figure:
    ex, ey = etiquetas.get(x, x), etiquetas.get(y, y)
    if y == "suscribe":
        if x in numericas:
            cortes = pd.qcut(df[x].rank(method="first"), q=10, labels=False)
            g = df.groupby(cortes).agg(cx=(x, "median"), tasa=("suscribe", "mean"), n=("suscribe", "size"))
            fig = go.Figure(go.Scatter(x=g["cx"], y=g["tasa"] * 100, mode="lines+markers", line=dict(color=AZUL, width=2),
                                       customdata=g["n"], hovertemplate=f"{ex} (mediana del decil): %{{x}}<br>"
                                       "%{y:.1f} % suscribe · %{customdata:,} clientes<extra></extra>"))
            fig.update_xaxes(title_text=f"{ex} (mediana de cada decil)")
        else:
            g = df.groupby(x).agg(tasa=("suscribe", "mean"), n=("suscribe", "size")).sort_values("tasa")
            fig = go.Figure(go.Bar(x=g.index.astype(str), y=g["tasa"] * 100, marker_color=AZUL, customdata=g["n"],
                                   hovertemplate="%{x}: %{y:.1f} % suscribe · %{customdata:,} clientes<extra></extra>"))
            fig.update_xaxes(title_text=ex, type="category")
        fig.update_yaxes(title_text="Clientes que suscriben (%)", rangemode="tozero")
        return fig.update_layout(title=f"Tasa de suscripción según {ex.lower()}", height=420, showlegend=False)

    if x not in numericas and y not in numericas:
        tabla = pd.crosstab(df[x], df[y], normalize="index") * 100
        fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns.astype(str), y=tabla.index.astype(str),
                                   colorscale=[[0, "#F4F5F7"], [1, AZUL]], colorbar=dict(title="%", thickness=12),
                                   text=np.round(tabla.values, 1), texttemplate="%{text}",
                                   hovertemplate=f"{ex} = %{{y}} · {ey} = %{{x}}: %{{z:.1f}} %<extra></extra>"))
        fig.update_xaxes(title_text=ey, type="category")
        fig.update_yaxes(title_text=ex, type="category")
        return fig.update_layout(title=f"{ey} dentro de cada categoría de {ex.lower()} (% por fila)", height=420)

    if x in numericas and y in numericas:
        muestra = df.sample(min(len(df), 4000), random_state=0)
        grupos = [(None, muestra)] if not color else list(muestra.groupby(color))
        fig = go.Figure()
        for i, (nombre, g) in enumerate(grupos):
            fig.add_trace(go.Scattergl(x=g[x], y=g[y], mode="markers", name=str(nombre) if nombre is not None else ey,
                                       marker=dict(size=6, opacity=0.45, color=PALETA[i % len(PALETA)]),
                                       hovertemplate=f"{ex}: %{{x}}<br>{ey}: %{{y}}<extra>%{{fullData.name}}</extra>"))
        fig.update_xaxes(title_text=ex)
        fig.update_yaxes(title_text=ey)
        titulo = f"{ey} frente a {ex.lower()} (muestra de {_num(len(muestra), 0)} clientes)"
    else:
        cat, num = (x, y) if x not in numericas else (y, x)
        fig = go.Figure()
        grupos = [(None, df)] if not color or color == cat else list(df.groupby(color))
        for i, (nombre, g) in enumerate(grupos):
            fig.add_trace(go.Box(x=g[cat].astype(str), y=g[num], name=str(nombre) if nombre is not None else etiquetas.get(num, num),
                                 marker_color=PALETA[i % len(PALETA)], boxpoints=False, line_width=1.5))
        fig.update_xaxes(title_text=etiquetas.get(cat, cat), type="category")
        fig.update_yaxes(title_text=etiquetas.get(num, num))
        fig.update_layout(boxmode="group")
        titulo = f"{etiquetas.get(num, num)} según {etiquetas.get(cat, cat).lower()}"
    return fig.update_layout(title=titulo, height=420, showlegend=bool(color))


def correlacion(df: pd.DataFrame, variables: list, etiquetas: dict) -> go.Figure:
    c = df[variables].corr(method="spearman")
    nombres = [etiquetas.get(v, v) for v in variables]
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, AZUL], [0.5, "#F4F5F7"], [1, NARANJA]],
                               text=np.round(c.values, 2), texttemplate="%{text}", textfont=dict(size=11),
                               colorbar=dict(title="ρ", thickness=12),
                               hovertemplate="%{y} · %{x}: ρ = %{z:.2f}<extra></extra>"))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title="Correlación de Spearman entre variables numéricas", height=520)


def faltantes(df: pd.DataFrame, categoricas: list, etiquetas: dict) -> go.Figure:
    pct = pd.Series({etiquetas.get(v, v): (df[v] == "unknown").mean() * 100 for v in categoricas})
    pct["Días desde el último contacto (999)"] = (df["pdays"] == 999).mean() * 100
    pct = pct[pct > 0].sort_values()
    fig = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h", marker_color=NARANJA,
                           text=[f"{_num(v)} %" for v in pct.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f} % de los registros<extra></extra>"))
    fig.update_xaxes(title_text="Registros con unknown o pdays = 999 (%)",
                     range=[0, max(pct.max() if len(pct) else 1, 1) * 1.18])
    return fig.update_layout(title="Valores faltantes (unknown y pdays = 999)", height=380, showlegend=False)


def atipicos(df: pd.DataFrame, numericas: list, etiquetas: dict) -> go.Figure:
    filas = {}
    for v in numericas:
        s = df[v] if v != "pdays" else df.loc[df[v] != 999, v]
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        riq = q3 - q1
        filas[etiquetas.get(v, v)] = ((s < q1 - 1.5 * riq) | (s > q3 + 1.5 * riq)).mean() * 100
    pct = pd.Series(filas).sort_values()
    fig = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h", marker_color=AZUL,
                           text=[f"{_num(v)} %" for v in pct.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f} % fuera de las cercas de Tukey<extra></extra>"))
    fig.update_xaxes(title_text="Valores fuera de Q1 − 1,5·RIQ y Q3 + 1,5·RIQ (%)",
                     range=[0, max(pct.max() if len(pct) else 1, 1) * 1.18])
    return fig.update_layout(title="Valores atípicos univariados (regla de Tukey)", height=380, showlegend=False)


# --------------------------------------------------------------------------------------------------------- Modelos
def confusion(fila: pd.Series) -> go.Figure:
    z = np.array([[fila["tn"], fila["fp"]], [fila["fn"], fila["tp"]]])
    texto = [[f"{_num(v, 0)}<br>({_num(v / z.sum() * 100)} %)" for v in r] for r in z]
    fig = go.Figure(go.Heatmap(z=z, x=["Predice: no", "Predice: sí"], y=["Real: no", "Real: sí"], text=texto,
                               texttemplate="%{text}", colorscale=[[0, "#F4F5F7"], [1, "#86B6EF"]], showscale=False,
                               hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title="Matriz de confusión (umbral por defecto)", height=330)


def roc(curvas: pd.DataFrame, auc: float) -> go.Figure:
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dot", width=1.5),
                                name="Azar", hoverinfo="skip"),
                     go.Scatter(x=curvas["x"], y=curvas["y"], mode="lines", line=dict(color=AZUL, width=2.5),
                                name=f"Modelo (AUC {_num(auc, 3)})",
                                hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")])
    fig.update_xaxes(title_text="Tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de verdaderos positivos (recall)", range=[0, 1.02])
    return fig.update_layout(title="Curva ROC en prueba", height=330)


def pr(curvas: pd.DataFrame, auc_pr: float, prevalencia: float) -> go.Figure:
    fig = go.Figure([go.Scatter(x=[0, 1], y=[prevalencia, prevalencia], mode="lines",
                                line=dict(color=GRIS, dash="dot", width=1.5), name=f"Azar ({_num(prevalencia, 3)})",
                                hoverinfo="skip"),
                     go.Scatter(x=curvas["x"], y=curvas["y"], mode="lines", line=dict(color=AZUL, width=2.5),
                                name=f"Modelo (AUC-PR {_num(auc_pr, 3)})",
                                hovertemplate="Recall %{x:.2f} · precisión %{y:.2f}<extra></extra>")])
    fig.update_xaxes(title_text="Recall", range=[0, 1])
    fig.update_yaxes(title_text="Precisión", range=[0, 1.02])
    return fig.update_layout(title="Curva precisión-recall en prueba", height=330)


def calibracion(cal: pd.DataFrame, brier: float, ece: float) -> go.Figure:
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dot", width=1.5),
                                name="Calibración perfecta", hoverinfo="skip"),
                     go.Scatter(x=cal["prob_media"], y=cal["frec_obs"], mode="lines+markers", name="Modelo",
                                line=dict(color=NARANJA, width=2.5), marker=dict(size=8, line=dict(color="white", width=1.5)),
                                customdata=cal["n"], hovertemplate="Probabilidad media %{x:.2f}<br>Frecuencia observada "
                                "%{y:.2f}<br>%{customdata:,} clientes<extra></extra>")])
    fig.update_xaxes(title_text="Probabilidad predicha (promedio del intervalo)", range=[0, 1])
    fig.update_yaxes(title_text="Frecuencia observada de suscripción", range=[0, 1.02])
    return fig.update_layout(title=f"Diagrama de confiabilidad · Brier {_num(brier, 3)} · ECE {_num(ece, 3)}", height=330)


def shap_global(df: pd.DataFrame, etiquetas: dict, n: int = 12) -> go.Figure:
    imp = df.groupby("variable")["phi"].agg(lambda s: s.abs().mean()).sort_values().tail(n)
    signo = df.groupby("variable")["phi"].mean().reindex(imp.index)
    fig = go.Figure(go.Bar(x=imp.values, y=[etiquetas.get(v, v) for v in imp.index], orientation="h", marker_color=AZUL,
                           customdata=signo.values, text=[_num(v, 3) for v in imp.values], textposition="outside",
                           cliponaxis=False,
                           hovertemplate="%{y}<br>Contribución media |SHAP|: %{x:.4f}<br>"
                                         "Contribución media con signo: %{customdata:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Contribución media |SHAP| (puntos de probabilidad)",
                     range=[0, max(imp.max(), 1e-6) * 1.15])
    return fig.update_layout(title="Importancia global: contribución media absoluta (SHAP)", height=420, showlegend=False)


def shap_local(phi: np.ndarray, nombres: list, valores: list, base: float, fx: float, n: int = 10) -> go.Figure:
    orden = np.argsort(np.abs(phi))[::-1]
    top, resto = orden[:n], orden[n:]
    etiquetas = [f"{nombres[i]} = {valores[i]}" for i in top]
    contrib = list(phi[top])
    if len(resto):
        etiquetas.append(f"Otras {len(resto)} variables")
        contrib.append(float(phi[resto].sum()))
    etiquetas, contrib = etiquetas[::-1], contrib[::-1]
    colores = [NARANJA if c > 0 else AZUL for c in contrib]
    fig = go.Figure(go.Bar(x=[c * 100 for c in contrib], y=etiquetas, orientation="h", marker_color=colores,
                           text=[("+" if c > 0 else "−") + _num(abs(c) * 100) + " pp" for c in contrib],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Contribución: %{x:.2f} puntos porcentuales<extra></extra>"))
    fig.add_vline(x=0, line_color="#C9CED6", line_width=1)
    extremo = max(abs(min(contrib)), abs(max(contrib)), 1e-6) * 100 * 1.3
    fig.update_xaxes(title_text="Contribución a la probabilidad (puntos porcentuales)",
                     range=[-extremo if min(contrib) < 0 else -extremo * 0.1, extremo if max(contrib) > 0 else extremo * 0.1])
    return fig.update_layout(title=f"Explicación local (SHAP): de {_num(base * 100)} % en promedio a {_num(fx * 100)} %",
                             height=440, showlegend=False)


def diferencia_critica(rangos: pd.DataFrame, destacado: str | None = None, subtitulo: str = "") -> go.Figure:
    """Diagrama de diferencia crítica (Nemenyi): rango promedio de cada modelo y la barra de la CD.

    ``destacado`` es el nombre del modelo que se resalta (el modelo original), siempre con el mismo color.
    """
    r = rangos.sort_values("rango").reset_index(drop=True)
    cd = float(r["cd"].iloc[0])
    k = len(r)
    colores = [NARANJA if m == destacado else AZUL for m in r["modelo"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 1 + cd], y=[k + 1.2, k + 1.2], mode="lines+text", line=dict(color=TINTA_2, width=6),
                             text=["", f"  CD = {_num(cd, 2)}"], textposition="middle right", hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[r["rango"].min(), r["rango"].max()], y=[0.3, 0.3], mode="lines",
                             line=dict(color=TINTA, width=6), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=r["rango"], y=np.arange(k, 0, -1), mode="markers+text",
                             marker=dict(size=13, color=colores, line=dict(color="white", width=2)),
                             text=[f"  {m} ({_num(v, 2)})" for m, v in zip(r["modelo"], r["rango"])],
                             textposition="middle right", hovertemplate="%{text}<extra></extra>", showlegend=False))
    for v, y, c in zip(r["rango"], np.arange(k, 0, -1), colores):
        fig.add_shape(type="line", x0=v, x1=v, y0=0.3, y1=y, line=dict(color=c, width=1, dash="dot"))
    fig.update_xaxes(title_text="Rango promedio (1 = mejor)", range=[0.8, k + 0.4], dtick=1)
    fig.update_yaxes(visible=False, range=[0, k + 1.8])
    titulo = "Diagrama de diferencia crítica (Nemenyi, α = 0,05)" + (f"<br><sup>{subtitulo}</sup>" if subtitulo else "")
    return fig.update_layout(title=titulo, height=440, margin=dict(l=10, r=10, t=70, b=50))


# -------------------------------------------------------------------------------------------- modelo original
def nivel_l2e(df: pd.DataFrame) -> go.Figure:
    """Tasa diaria observada frente a la probabilidad del L2E para el cliente promedio (nivel del día + pendientes)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["dia"], y=df["tasa"] * 100, mode="markers", name="Tasa observada del día",
        marker=dict(size=np.clip(np.sqrt(df["n"]) * 0.9, 4, 14), color=GRIS, opacity=0.45, line=dict(width=0)),
        customdata=np.c_[df["periodo"], df["n"]],
        hovertemplate="Día %{x} (%{customdata[0]})<br>Tasa observada: %{y:.1f} %<br>%{customdata[1]:,} llamadas"
                      "<extra></extra>"))
    for conjunto, guion in (("Entrenamiento (suavizado)", "dot"), ("Prueba (filtrado, un día de demora)", "solid")):
        s = df[df["conjunto"] == conjunto]
        fig.add_trace(go.Scatter(x=s["dia"], y=s["prob"] * 100, mode="lines", name=f"L2E · {conjunto.lower()}",
                                 line=dict(color=NARANJA, width=2, dash=guion), customdata=s["periodo"],
                                 hovertemplate="Día %{x} (%{customdata})<br>L2E: %{y:.1f} %<extra></extra>"))
    inicio = df.loc[df["conjunto"] != "Entrenamiento (suavizado)", "dia"].min()
    fig.add_vline(x=inicio - 0.5, line=dict(color=TINTA_2, width=1, dash="dash"))
    fig.add_annotation(x=inicio, y=0.98, yref="paper", text="  inicio de la prueba", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(size=12, color=TINTA_2))
    fig.update_yaxes(title_text="Clientes que suscriben (%)", rangemode="tozero")
    fig.update_xaxes(title_text="Día de contacto (orden temporal)")
    # Leyenda al pie del contenedor: con tres entradas largas se parte en varias filas en pantallas angostas, y arriba
    # se montaba sobre el título; anclada al contenedor, Plotly le reserva su espacio debajo del eje
    return fig.update_layout(title="El nivel diario que sigue el L2E", height=400, hovermode="closest",
                             margin=dict(l=10, r=10, t=56, b=10), legend=dict(yref="container", y=0, yanchor="bottom"))


def convergencia(df: pd.DataFrame) -> go.Figure:
    """Puntaje de cada configuración evaluada por Optuna y mejor puntaje acumulado (curva de convergencia)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["n"], y=df["puntaje"], mode="markers", name="Configuración evaluada",
                             marker=dict(size=9, color=GRIS, line=dict(color="white", width=1.5)),
                             hovertemplate="Evaluación %{x}: AUC-PR %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["n"], y=df["mejor"], mode="lines", name="Mejor hasta el momento",
                             line=dict(color=NARANJA, width=2, shape="hv"),
                             hovertemplate="Tras %{x} evaluaciones: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Configuraciones evaluadas", dtick=2)
    fig.update_yaxes(title_text="AUC-PR de la validación interna")
    return fig.update_layout(title="Convergencia de la optimización (Optuna TPE, modelo final)", height=360,
                             margin=dict(l=10, r=10, t=84, b=10))


def aprendizaje(df: pd.DataFrame) -> go.Figure:
    """AUC-PR de prueba según el tamaño del entrenamiento (los registros más recientes)."""
    fig = go.Figure()
    for modelo, color in (("L2E", NARANJA), ("Referencia E2", AZUL)):
        s = df[df["modelo"] == modelo].sort_values("registros")
        fig.add_trace(go.Scatter(x=s["registros"], y=s["auc_pr"], mode="lines+markers",
                                 name="L2E (modelo original)" if modelo == "L2E" else "Referencia E2 (logística C = 0,01)",
                                 line=dict(color=color, width=2), marker=dict(size=9, color=color,
                                                                               line=dict(color="white", width=1.5)),
                                 hovertemplate="%{x:,} registros: AUC-PR %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Registros de entrenamiento (los más recientes)", tickformat=",d")
    fig.update_yaxes(title_text="AUC-PR en la prueba")
    return fig.update_layout(title="Curva de aprendizaje", height=360, margin=dict(l=10, r=10, t=84, b=10))
