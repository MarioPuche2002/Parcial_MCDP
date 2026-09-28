"""API del detector.  Local: uvicorn src.api:app --reload  ->  http://localhost:8000/docs

Dos validaciones en la puerta:
1. BaseModel: estructura (campos completos, tipos correctos, minimo 10 lecturas).
2. Pandera: valores fisicos, con el MISMO contrato del entrenamiento. Las lecturas
   invalidas se quitan; si quedan menos de 10 validas, se responde error.
"""
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predecir
from src.schema import esquema_lecturas

MIN_LECTURAS = 10  # con menos datos la desviacion no es confiable

app = FastAPI(title="Detector de fallas de GPU L40")


class Lectura(BaseModel):
    temp_c: float
    power_w: float
    util_pct: float
    clock_mhz: float
    ecc_errors: int


class Ventana(BaseModel):
    lecturas: list[Lectura] = Field(min_length=MIN_LECTURAS)


@app.get("/")
def salud():
    return {"estado": "ok"}


@app.post("/predecir")
def predecir_ventana(ventana: Ventana):
    df = pd.DataFrame([l.model_dump() for l in ventana.lecturas])

    # mismo contrato que en el entrenamiento: quita lecturas con valores imposibles
    df_valido = esquema_lecturas.validate(df, lazy=True)

    if len(df_valido) < MIN_LECTURAS:
        raise HTTPException(
            status_code=422,
            detail=f"Solo {len(df_valido)} de {len(df)} lecturas tienen valores validos; "
                   f"se necesitan al menos {MIN_LECTURAS}. Revise rangos y unidades (temp_c en Celsius).",
        )

    return predecir(df_valido.to_dict("records"))