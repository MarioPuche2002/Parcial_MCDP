"""Calculo de features. Lo usan data.py Y api.py (mismas features en los dos lados)."""
import pandas as pd

SENALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
TAM_VENTANA = 30


def calcular_features(ventana):
    feats = {}
    for col in SENALES:
        feats[f"{col}_media"] = ventana[col].mean()
        feats[f"{col}_mediana"] = ventana[col].median()
        feats[f"{col}_std"] = ventana[col].std()
        feats[f"{col}_min"] = ventana[col].min()
        feats[f"{col}_max"] = ventana[col].max()
    return feats


def ventanear(df, tam=TAM_VENTANA):
    """Parte cada episodio en ventanas de `tam` segundos (no mezcla episodios)."""
    filas = []
    for (episodio, ventana_id), ventana in df.groupby(["episodio_id", df["segundo"] // tam]):
        fila = calcular_features(ventana)
        fila["estado"] = ventana["estado"].iloc[0]
        fila["episodio_id"] = episodio
        fila["ventana"] = ventana_id
        filas.append(fila)
    return pd.DataFrame(filas)