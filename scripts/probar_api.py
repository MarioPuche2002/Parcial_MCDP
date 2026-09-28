"""Prueba la API corriendo (local o en Docker).  Uso: python scripts/probar_api.py"""
import pandas as pd
import requests

URL = "http://localhost:8000/predecir"
SENALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]

df = pd.read_csv("data/telemetria_publica.csv")

# una ventana de 30 lecturas de cada estado
for estado in df["estado"].unique():
    episodio = df[df["estado"] == estado]["episodio_id"].iloc[0]
    lecturas = df[df["episodio_id"] == episodio][SENALES].head(30).to_dict("records")
    print(estado, "->", requests.post(URL, json={"lecturas": lecturas}).json())

# un episodio completo (300 lecturas) como una sola ventana
completo = df[df["episodio_id"] == 36][SENALES].to_dict("records")
print("Episodio completo (300 lecturas) ->", requests.post(URL, json={"lecturas": completo}).json())

# ventana muy corta: debe ser rechazada
print("Ventana de 3 lecturas ->", requests.post(URL, json={"lecturas": completo[:3]}).status_code)