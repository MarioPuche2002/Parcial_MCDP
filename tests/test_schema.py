"""Pruebas del contrato de datos (pandera).
Correr desde la raiz:  python -m pytest -v
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.schema import esquema


def datos_validos():
    """10 segundos de telemetria sana, un solo episodio."""
    return pd.DataFrame({
        "episodio_id": [0] * 10,
        "segundo": list(range(10)),
        "temp_c": [55.0] * 10,
        "power_w": [200.0] * 10,
        "util_pct": [70.0] * 10,
        "clock_mhz": [2400.0] * 10,
        "ecc_errors": [0] * 10,
        "estado": ["normal"] * 10,
    })


def test_datos_validos_pasan_completos():
    df = datos_validos()
    assert len(esquema.validate(df, lazy=True)) == 10


def test_valor_faltante_se_descarta():
    df = datos_validos()
    df.loc[3, "temp_c"] = np.nan
    assert len(esquema.validate(df, lazy=True)) == 9


def test_potencia_negativa_se_descarta():
    df = datos_validos()
    df.loc[3, "power_w"] = -13.5
    assert len(esquema.validate(df, lazy=True)) == 9


def test_ecc_negativo_se_descarta():
    df = datos_validos()
    df.loc[3, "ecc_errors"] = -1
    assert len(esquema.validate(df, lazy=True)) == 9


def test_estado_invalido_se_descarta():
    df = datos_validos()
    df.loc[3, "estado"] = "roto"
    assert len(esquema.validate(df, lazy=True)) == 9


def test_fahrenheit_se_descarta():
    df = datos_validos()
    df["temp_c"] = df["temp_c"] * 9 / 5 + 32  # 55 C -> 131 F
    assert len(esquema.validate(df, lazy=True)) == 0


def test_columna_faltante_da_error():
    df = datos_validos().drop(columns="util_pct")
    with pytest.raises(pa.errors.SchemaErrors):
        esquema.validate(df, lazy=True)


def test_columna_extra_da_error():
    df = datos_validos()
    df["columna_rara"] = 1
    with pytest.raises(pa.errors.SchemaErrors):
        esquema.validate(df, lazy=True)


def test_datos_reales_descartan_3_filas():
    df = pd.read_csv("data/telemetria_publica.csv")
    assert len(df) - len(esquema.validate(df, lazy=True)) == 3