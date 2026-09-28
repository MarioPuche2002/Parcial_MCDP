"""Prediccion de una ventana. La usa api.py; tambien se puede probar sin levantar la API.
"""
import joblib
import pandas as pd

from src.features import SENALES, calcular_features

modelo = joblib.load("models/modelo.joblib")


def predecir(lecturas):
    """Recibe una lista de lecturas (dicts con las 5 senales) y devuelve el estado."""
    df = pd.DataFrame(lecturas)
    X = pd.DataFrame([calcular_features(df)])
    probas = modelo.predict_proba(X)[0]
    return {
        "estado_predicho": modelo.classes_[probas.argmax()],
        "confianza": round(float(probas.max()), 3),
    }


if __name__ == "__main__":
    # prueba rapida: 30 segundos de un episodio de cada estado
    df = pd.read_csv("data/telemetria_publica.csv")
    for estado in df["estado"].unique():
        episodio = df[df["estado"] == estado]["episodio_id"].iloc[0]
        lecturas = df[df["episodio_id"] == episodio][SENALES].head(30).to_dict("records")
        print(estado, "->", predecir(lecturas))