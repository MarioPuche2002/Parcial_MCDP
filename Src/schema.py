"""Contrato de datos de la telemetria (pandera)."""
import pandera.pandas as pa

ESTADOS = ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]

esquema = pa.DataFrameSchema(
    {
        "episodio_id": pa.Column(int, pa.Check.ge(0)),
        "segundo": pa.Column(int, pa.Check.ge(0)),
        "temp_c": pa.Column(float, pa.Check.in_range(0, 110)), 
        "power_w": pa.Column(float, [pa.Check.gt(0), pa.Check.le(400)]),
        "util_pct": pa.Column(float, pa.Check.in_range(0, 100)),
        "clock_mhz": pa.Column(float, pa.Check.in_range(0, 3000)),
        "ecc_errors": pa.Column(int, pa.Check.ge(0)),
        "estado": pa.Column(str, pa.Check.isin(ESTADOS)),
    },
    strict=True,             # ni columnas extra ni faltantes
    drop_invalid_rows=True,  # las filas que violan el contrato se descartan
)