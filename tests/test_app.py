"""La aplicación arranca, registra las tres páginas y sus funciones responden con entradas válidas e inválidas."""
import dash
import pytest

import app  # noqa: F401  crea la aplicación y registra las páginas


def test_tres_paginas_registradas():
    rutas = {p["path"] for p in dash.page_registry.values()}
    assert {"/", "/eda", "/modelos"} <= rutas


@pytest.mark.parametrize("ruta", ["/", "/eda", "/modelos"])
def test_layouts_se_construyen(ruta):
    pagina = next(p for p in dash.page_registry.values() if p["path"] == ruta)
    layout = pagina["layout"]
    assert (layout() if callable(layout) else layout) is not None


def test_validacion_del_formulario():
    from pages import modelos as pm
    from src.data import esquema

    correctos = {v["variable"]: v["defecto"] for v in esquema()["variables"]}
    limpio, errores = pm.validar(correctos)
    assert not errores and set(limpio) == set(correctos)

    _, errores = pm.validar({**correctos, "age": 150, "pdays": 500, "job": "astronauta"})
    assert len(errores) == 3


def test_filtros_del_eda():
    from pages import eda

    df = eda.banco()
    salida = eda.actualizar_generales(df["fecha"].min(), df["fecha"].max(), [17, 98], "Prueba", ["cellular"])
    assert "registros seleccionados" in salida[0]
    vacio = eda.actualizar_generales(df["fecha"].min(), df["fecha"].max(), [17, 18], "Prueba", ["telephone"])
    assert vacio[0].startswith("0 ") or "registros seleccionados" in vacio[0]


def _en_contexto(funcion, *args, salidas=None, estados=None):
    """Ejecuta un callback con un contexto de Dash simulado (outputs_list y states_list)."""
    from contextvars import copy_context

    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def correr():
        context_value.set(AttributeDict(outputs_list=salidas or [], states_list=estados or [], triggered_inputs=[]))
        return funcion(*args)

    return copy_context().run(correr)


def test_cliente_al_azar_y_prediccion_en_los_ocho_modelos():
    import dash_bootstrap_components as dbc
    from pages import modelos as pm
    from src.data import esquema, modelos

    ids = [{"id": {"tipo": "entrada", "var": v["variable"]}, "property": "value"} for v in esquema()["variables"]]
    valores = _en_contexto(pm.cargar_azar, 3, salidas=ids)
    assert len(valores) == len(ids) and all(v is not None for v in valores)

    estados = [[{**i, "value": v} for i, v in zip(ids, valores)]]
    for m in modelos():
        tarjeta, figura = _en_contexto(pm.predecir_cliente, None, m, valores, estados=estados)
        assert not isinstance(tarjeta, dbc.Alert), f"{m}: la predicción devolvió un error"
        assert figura.layout.title.text.startswith("Explicación local")


def test_prediccion_con_datos_invalidos_muestra_alerta():
    import dash_bootstrap_components as dbc
    from pages import modelos as pm
    from src.data import esquema

    estados = [[{"id": {"tipo": "entrada", "var": v["variable"]}, "property": "value",
                 "value": (150 if v["variable"] == "age" else v["defecto"])} for v in esquema()["variables"]]]
    tarjeta, _ = _en_contexto(pm.predecir_cliente, 1, "naive_bayes", None, estados=estados)
    assert isinstance(tarjeta, dbc.Alert) and tarjeta.color == "danger"


def test_diagnostico_de_cada_modelo():
    from pages import modelos as pm
    from src.data import modelos

    for m in modelos():
        figuras = pm.diagnostico(m)
        assert len(figuras) == 7


def test_diagrama_de_diferencia_critica_con_el_modelo_original():
    from pages import modelos as pm

    for bloques in pm.BLOQUES_CD:
        figura = pm.diagrama_cd(bloques)
        etiquetas = " ".join(figura.data[2].text)
        assert "L2E (modelo original)" in etiquetas, bloques


def test_prediccion_del_l2e_usa_el_dia_siguiente_al_ultimo_cerrado():
    from src.data import entrada_modelo, esquema, fila_entrada, modelos, probabilidad

    X = fila_entrada({v["variable"]: v["defecto"] for v in esquema()["variables"]})
    meta = modelos()["l2e"]["meta"]
    assert entrada_modelo("l2e", X)["dia"].iloc[0] == meta["dia_prediccion"]
    assert "dia" not in entrada_modelo("naive_bayes", X).columns
    assert 0 < probabilidad("l2e", X)[0] < 1
