"""Define el modelo ganador del notebook (sin entrenarlo)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def crear_modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))