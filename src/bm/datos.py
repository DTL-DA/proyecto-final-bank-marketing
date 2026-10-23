"""Carga de datos, reconstrucción del calendario y partición cronológica."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config as cfg


def cargar_datos(ruta=cfg.RUTA_DATOS) -> pd.DataFrame:
    """Lee ``bank-additional-full.csv`` (separador ``;``) conservando el orden temporal de las filas."""
    return pd.read_csv(ruta, sep=";")


def reconstruir_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye año, mes, periodo y día de contacto a partir del orden de las filas.

    El archivo viene ordenado de mayo de 2008 a noviembre de 2010 y solo trae el mes; cada vez que el número de mes
    retrocede empieza un año nuevo. Un día de contacto es un bloque contiguo con el mismo periodo y día de la semana.

    Parameters
    ----------
    df : pandas.DataFrame
        Datos en el orden original del archivo.

    Returns
    -------
    pandas.DataFrame
        Columnas ``anio``, ``mes``, ``periodo`` (``AAAA-MM``) y ``dia`` (identificador creciente), con el índice de ``df``.
    """
    mes = df["month"].map({m: i + 1 for i, m in enumerate(cfg.ORDEN_MESES)}).to_numpy()
    anio = 2008 + np.r_[0, np.cumsum(np.diff(mes) < 0)]
    tiempo = pd.DataFrame({"anio": anio, "mes": mes}, index=df.index)
    tiempo["periodo"] = [f"{a}-{m:02d}" for a, m in zip(anio, mes)]
    clave = tiempo["periodo"] + "-" + df["day_of_week"]
    tiempo["dia"] = (clave != clave.shift()).cumsum()
    return tiempo


@dataclass
class Particion:
    """Partición cronológica 80/20 y objetos asociados."""
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series
    r_tr: pd.Series
    r_te: pd.Series
    tiempo_tr: pd.DataFrame
    tiempo_te: pd.DataFrame
    gap: int


def preparar_particion(df: pd.DataFrame | None = None) -> Particion:
    """Elimina duplicados exactos, reconstruye el calendario y reserva el último 20 % como prueba.

    Returns
    -------
    Particion
        ``y`` es el objetivo binario (1 = suscribe) y ``r`` el objetivo continuo de la rama de regresión,
        ``log1p(duration)``. ``X`` no incluye ``duration`` ni ``y``. ``gap`` es el percentil 95 de los registros
        por día de contacto en entrenamiento, usado como separación en ``TimeSeriesSplit``.
    """
    df = cargar_datos() if df is None else df
    tiempo = reconstruir_tiempo(df)
    limpio = df.drop_duplicates(keep="first")
    tiempo = tiempo.loc[limpio.index]
    y = limpio[cfg.OBJETIVO].eq("yes").astype(int)
    r = np.log1p(limpio[cfg.OBJETIVO_REG].astype(float)).rename("log_duration")
    X = limpio.drop(columns=[cfg.OBJETIVO])
    n_te = int(np.ceil(cfg.FRACCION_PRUEBA * len(X)))
    n_tr = len(X) - n_te
    t_tr, t_te = tiempo.iloc[:n_tr], tiempo.iloc[n_tr:]
    gap = int(np.ceil(t_tr["dia"].value_counts().quantile(0.95)))
    return Particion(X.iloc[:n_tr], X.iloc[n_tr:], y.iloc[:n_tr], y.iloc[n_tr:], r.iloc[:n_tr], r.iloc[n_tr:],
                     t_tr, t_te, gap)


def cv_externa(gap: int) -> TimeSeriesSplit:
    """Bucle externo de la validación anidada: ``TimeSeriesSplit`` de ``K_EXTERNO`` pliegues con separación ``gap``."""
    return TimeSeriesSplit(n_splits=cfg.K_EXTERNO, gap=gap)


def cv_interna(gap: int) -> TimeSeriesSplit:
    """Bucle interno: ``TimeSeriesSplit`` de ``K_INTERNO`` pliegues sobre el entrenamiento de cada pliegue externo."""
    return TimeSeriesSplit(n_splits=cfg.K_INTERNO, gap=gap)
