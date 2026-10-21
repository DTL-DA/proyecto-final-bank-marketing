"""Artefactos de los modelos: prueba de ida y vuelta y metadatos (bloque B de la guía)."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import ARTEFACTOS, DATOS

MODELOS = sorted(p.name for p in ARTEFACTOS.iterdir() if p.is_dir())
IDA_VUELTA = pd.read_parquet(DATOS / "roundtrip.parquet")


def test_estan_los_siete_modelos():
    assert MODELOS == sorted(["arbol", "bosque", "knn", "logistica", "naive_bayes", "svm", "xgboost"])


@pytest.mark.parametrize("modelo", MODELOS)
def test_ida_y_vuelta(modelo):
    """El modelo serializado reproduce, sobre 400 clientes fijos de la prueba, las predicciones del modelo exportado."""
    pipe = joblib.load(ARTEFACTOS / modelo / "modelo.joblib")
    meta = json.loads((ARTEFACTOS / modelo / "metadata.json").read_text(encoding="utf-8"))
    X = IDA_VUELTA[meta["columnas_entrada"]]
    puntaje = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)
    np.testing.assert_allclose(puntaje, IDA_VUELTA[f"esperado__{modelo}"].to_numpy(), rtol=0, atol=1e-9)
    assert "bm" not in sys.modules, "los artefactos no deben depender del paquete del cuaderno"


@pytest.mark.parametrize("modelo", MODELOS)
def test_metadatos_completos(modelo):
    meta = json.loads((ARTEFACTOS / modelo / "metadata.json").read_text(encoding="utf-8"))
    for clave in ("versiones", "semilla", "hiperparametros", "columnas_entrada", "variables_usadas",
                  "metricas_bucle_externo", "metricas_prueba", "motivo_seleccion"):
        assert clave in meta, clave
    assert meta["semilla"] == 42
    assert {"media", "de"} <= set(meta["metricas_bucle_externo"]["auc_pr"])
    assert {"sklearn", "xgboost", "imblearn", "numpy", "pandas"} <= set(meta["versiones"])
    assert (DATOS / "esquema.json").exists()
