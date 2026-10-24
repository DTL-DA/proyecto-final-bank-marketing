"""Carga y caché de datos, artefactos y resultados precalculados.

Todo se lee una sola vez (``functools.lru_cache``); el dashboard nunca entrena ni optimiza modelos, solo usa los
artefactos de ``artifacts/`` y los resultados de ``data/``.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"
ARTEFACTOS = RAIZ / "artifacts"


@lru_cache(maxsize=1)
def banco() -> pd.DataFrame:
    """Registros sin duplicados exactos, con periodo reconstruido y conjunto (entrenamiento o prueba)."""
    df = pd.read_parquet(DATOS / "bank.parquet")
    df["suscribe"] = (df["y"] == "yes").astype(int)
    return df


@lru_cache(maxsize=1)
def esquema() -> dict:
    """Variables de entrada del modelo: tipo, rango válido, categorías y valor por defecto."""
    with open(DATOS / "esquema.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def contexto() -> dict:
    with open(DATOS / "contexto.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def resultado(nombre: str) -> pd.DataFrame:
    """Tabla precalculada de ``data/resultados`` (tabla de líderes, curvas, calibración, SHAP global, etc.)."""
    return pd.read_parquet(DATOS / "resultados" / f"{nombre}.parquet")


@lru_cache(maxsize=1)
def fondo() -> pd.DataFrame:
    """Filas de referencia del entrenamiento para las explicaciones SHAP locales."""
    return pd.read_parquet(DATOS / "fondo_shap.parquet")


@lru_cache(maxsize=None)
def resumen(nombre: str) -> dict:
    """Resumen precalculado en JSON de ``data/resultados`` (cifras de la subsección del modelo original)."""
    with open(DATOS / "resultados" / f"{nombre}.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def modelos() -> dict:
    """Pipelines serializados y sus metadatos, en el orden de la tabla de líderes.

    Cada modelo usa ``artifacts/<modelo>/modelo.joblib``, salvo que su ``metadata.json`` indique otro archivo: el L2E
    carga ``modelo_vigente.joblib``, que ya asimiló las etiquetas de todos los días cerrados.
    """
    salida = {}
    for m in resultado("leaderboard")["modelo"]:
        with open(ARTEFACTOS / m / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        salida[m] = {"pipeline": joblib.load(ARTEFACTOS / m / meta.get("archivo", "modelo.joblib")), "meta": meta}
    return salida


def opciones_modelo() -> list[dict]:
    lb = resultado("leaderboard")
    return [{"label": r.nombre if r.modelo == "l2e" else f"{r.nombre} · {r.balanceo} · {r.optimizador}",
             "value": r.modelo} for r in lb.itertuples()]


def entrada_modelo(m: str, X: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas propias de un modelo: el L2E necesita el día de contacto (el siguiente al último cerrado)."""
    dia = modelos()[m]["meta"].get("dia_prediccion")
    return X if dia is None or "dia" in X.columns else X.assign(dia=dia)


def puntaje_crudo(m: str, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad del modelo o, en la SVM lineal, su función de decisión."""
    pipe = modelos()[m]["pipeline"]
    X = entrada_modelo(m, X)
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


def probabilidad(m: str, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de suscribir; en la SVM se aplica el escalado de Platt guardado en sus metadatos."""
    s = puntaje_crudo(m, X)
    platt = modelos()[m]["meta"]["platt"]
    return s if platt is None else 1 / (1 + np.exp(-(platt["a"] * s + platt["b"])))


def predecir(m: str, X: pd.DataFrame) -> np.ndarray:
    return modelos()[m]["pipeline"].predict(entrada_modelo(m, X))


def percentil(m: str, puntaje: float) -> float:
    """Proporción de clientes de prueba con un puntaje menor que el dado (posición en el ranking de llamadas)."""
    return float((resultado("puntajes_prueba")[m].to_numpy() < puntaje).mean())


def fila_entrada(valores: dict) -> pd.DataFrame:
    """Arma una fila con todas las columnas de entrada del modelo y los mismos tipos del entrenamiento.

    Las columnas que el modelo no usa (``duration`` y las macro retiradas por VIF) se llenan con la mediana del
    entrenamiento; el preprocesador las descarta, así que no cambian la predicción.
    """
    e = esquema()
    completa = {**e["no_usadas"], **valores}
    fila = pd.DataFrame([{c: completa[c] for c in e["columnas_entrada"]}])
    return fila.astype(fondo().dtypes.to_dict())
