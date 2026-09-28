"""Pruebas de la API sin levantar el servidor.
Correr desde la raiz (con models/modelo.joblib ya entrenado):  python -m pytest -v
"""
from fastapi.testclient import TestClient

from src.api import app

cliente = TestClient(app)


def ventana(n=10):
    """Arma n lecturas de sobrecalentamiento, con algo de variacion."""
    lecturas = []
    for i in range(n):
        lecturas.append({
            "temp_c": 88.0 + i % 3,
            "power_w": 290.0,
            "util_pct": 95.0,
            "clock_mhz": 2100 + 20 * (i % 4),
            "ecc_errors": 0,
        })
    return {"lecturas": lecturas}


def test_ventana_valida_predice():
    r = cliente.post("/predecir", json=ventana())
    assert r.status_code == 200
    assert r.json()["estado_predicho"] == "sobrecalentamiento"


def test_menos_de_10_lecturas_da_422():
    r = cliente.post("/predecir", json=ventana(n=3))
    assert r.status_code == 422


def test_una_lectura_invalida_se_quita_y_predice():
    datos = ventana(n=30)
    datos["lecturas"][0]["power_w"] = -13.5  # pandera la quita, quedan 29 validas
    r = cliente.post("/predecir", json=datos)
    assert r.status_code == 200


def test_fahrenheit_da_422():
    datos = ventana()
    for lectura in datos["lecturas"]:
        lectura["temp_c"] = lectura["temp_c"] * 9 / 5 + 32  # 88 C -> 190 F
    r = cliente.post("/predecir", json=datos)
    assert r.status_code == 422


def test_campo_faltante_da_422():
    datos = ventana()
    del datos["lecturas"][0]["temp_c"]
    r = cliente.post("/predecir", json=datos)
    assert r.status_code == 422