from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predecir

app = FastAPI(title="Detector de fallas de GPU L40")


class Lectura(BaseModel):
    temp_c: float = Field(ge=0, le=110)
    power_w: float = Field(ge=-50, le=400)  # margen: en falla_alimentacion el sensor a veces da negativos
    util_pct: float = Field(ge=0, le=100)
    clock_mhz: float = Field(ge=0, le=3000)
    ecc_errors: int = Field(ge=0)


class Ventana(BaseModel):
    lecturas: list[Lectura] = Field(min_length=10)  # con menos datos la desviacion no es confiable


@app.get("/")
def salud():
    return {"estado": "ok"}


@app.post("/predecir")
def predecir_ventana(ventana: Ventana):
    return predecir([l.model_dump() for l in ventana.lecturas])