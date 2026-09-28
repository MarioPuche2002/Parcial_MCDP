"""Valida la telemetria y genera el CSV de entrenamiento.
Uso (desde la raiz):  python -m src.data
"""
import pandas as pd

from src.features import ventanear
from src.schema import esquema

# 1. cargar y validar con el contrato
df = pd.read_csv("data/telemetria_publica.csv")
df_valido = esquema.validate(df, lazy=True)
descartadas = len(df) - len(df_valido)
print("Filas descartadas por el contrato:", descartadas)
assert descartadas < 0.01 * len(df), "Mas del 1% de filas invalidas: revisar datos (unidades?)"

# 2. ventanas -> features -> csv
features_train = ventanear(df_valido)
features_train.to_csv("data/features_train.csv", index=False)
print("features_train.csv:", features_train.shape)