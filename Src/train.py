"""Entrena el modelo y guarda el pipeline completo.
Uso (desde la raiz, despues de python -m src.data):  python -m src.train
"""
import joblib
import pandas as pd
from src.model import crear_modelo

datos = pd.read_csv("data/features_train.csv")
X = datos.drop(columns=["estado", "episodio_id", "ventana"])  # identificadores y etiqueta no son features
y = datos["estado"]
print("Ventanas de entrenamiento:", X.shape)

modelo = crear_modelo()
modelo.fit(X, y)
joblib.dump(modelo, "models/modelo.joblib")
print("Modelo guardado en models/modelo.joblib")