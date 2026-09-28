# Detector de fallas de GPU (NVIDIA L40)

API que recibe una ventana de telemetría de una GPU y devuelve su estado: `normal`, `sobrecalentamiento`, `degradacion_memoria` o `falla_alimentacion`.

```
telemetria -> contrato pandera -> ventanas de 30 s -> 25 features -> regresion logistica -> API FastAPI -> Docker
```

## Correr con Docker

```bash
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

Abrir http://localhost:8000/docs y probar `POST /predecir`.

> En Windows sin Docker Desktop, estos dos comandos se corren desde WSL (Ubuntu): `sudo service docker start` y `cd /mnt/c/.../Parcial_MCDP` antes del build.

### Ejemplo de petición

La ventana debe tener **mínimo 10 lecturas válidas**, cada una con los 5 campos:

```json
{
  "lecturas": [
    {"temp_c": 88.2, "power_w": 301.0, "util_pct": 92.3, "clock_mhz": 2136, "ecc_errors": 0},
    {"temp_c": 91.4, "power_w": 293.4, "util_pct": 91.7, "clock_mhz": 2218, "ecc_errors": 0},
    {"temp_c": 83.1, "power_w": 281.9, "util_pct": 90.4, "clock_mhz": 2075, "ecc_errors": 0},
    {"temp_c": 87.7, "power_w": 268.2, "util_pct": 98.3, "clock_mhz": 2149, "ecc_errors": 1},
    {"temp_c": 88.4, "power_w": 282.3, "util_pct": 99.8, "clock_mhz": 2105, "ecc_errors": 0},
    {"temp_c": 89.8, "power_w": 276.6, "util_pct": 94.0, "clock_mhz": 2093, "ecc_errors": 0},
    {"temp_c": 82.9, "power_w": 282.7, "util_pct": 94.9, "clock_mhz": 2205, "ecc_errors": 1},
    {"temp_c": 87.2, "power_w": 292.4, "util_pct": 90.5, "clock_mhz": 2267, "ecc_errors": 0},
    {"temp_c": 89.2, "power_w": 290.8, "util_pct": 95.3, "clock_mhz": 2065, "ecc_errors": 0},
    {"temp_c": 89.4, "power_w": 308.6, "util_pct": 95.9, "clock_mhz": 2167, "ecc_errors": 0}
  ]
}
```

Respuesta:

```json
{"estado_predicho": "sobrecalentamiento", "confianza": 0.997}
```

## Correr local (desarrollo)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -v                 # 1. tests del contrato pandera y de la API
python -m src.data                  # 2. valida y genera data/features_train.csv
python -m src.evaluate              # 3. evaluacion por episodio (sin fuga)
python -m src.train                 # 4. entrena y guarda models/modelo.joblib
python -m src.predict               # 5. prueba rapida del modelo sin API
uvicorn src.api:app --reload        # 6. levanta la API
python scripts/probar_api.py        # 7. prueba la API (en otra terminal)
```

## Estructura

```
src/schema.py          contratos pandera: entrenamiento (esquema) y API (esquema_lecturas)
src/features.py        ventaneo y features por ventana (lo usan data.py y predict.py)
src/data.py            valida la telemetria y genera features_train.csv
src/model.py           define el modelo: StandardScaler + regresion logistica
src/evaluate.py        validacion cruzada por episodio y matriz de confusion
src/train.py           entrena y guarda el pipeline completo
src/predict.py         predice el estado de una ventana
src/api.py             API FastAPI: valida con BaseModel y pandera, luego predice
models/modelo.joblib   pipeline ya entrenado
tests/test_schema.py   tests del contrato de entrenamiento
tests/test_api.py      tests de la API (sin levantar el servidor)
notebooks/EDA.ipynb    EDA y prueba de modelos
scripts/probar_api.py  prueba de la API corriendo
data/                  telemetria publica y csv generados
reports/               graficas
```

## EDA

**Los datos:** 14.400 filas, una por segundo: 48 episodios de 300 segundos, 12 por estado (clases balanceadas), sin nulos.

**Calidad de datos:** el CSV original traía 3 lecturas con potencia negativa (episodios 36, 38 y 44, todos de `falla_alimentacion`). Son físicamente imposibles, así que el **contrato pandera las descartó** antes de ventanear. El modelo se entrenó con 14.397 filas válidas.

**Una fila por episodio:** cada episodio se resumió en una fila con la media, mediana y desviación de cada señal (`data/resumen_episodios.csv`, 48 filas). Este CSV es para analizar; el modelo se entrena con ventanas. Promedio por estado:

| Señal | Estadístico | normal | sobrecalentamiento | degradacion_memoria | falla_alimentacion |
|---|---|---|---|---|---|
| temp_c | media | 54.9 | **88.0** | 60.0 | 50.0 |
| | mediana | 54.9 | 88.1 | 59.9 | 50.0 |
| | desviación | 4.0 | 3.0 | 5.0 | 8.0 |
| power_w | media | 200.1 | 290.3 | 209.1 | 140.6 |
| | mediana | 200.0 | 290.4 | 209.3 | 140.0 |
| | desviación | 20.0 | 10.1 | 25.1 | **44.8** |
| util_pct | media | 69.8 | 94.9 | 67.8 | 54.5 |
| | mediana | 69.4 | 95.1 | 67.9 | 54.2 |
| | desviación | 12.0 | 3.7 | 13.9 | 24.0 |
| clock_mhz | media | 2400.7 | 2098.6 | 2378.9 | 1798.6 |
| | mediana | 2400.5 | 2097.8 | 2379.4 | 1798.9 |
| | desviación | 29.7 | 80.4 | 39.8 | **204.5** |
| ecc_errors | media | 0.0 | 0.1 | **5.0** | 0.2 |
| | mediana | 0.0 | 0.0 | 5.0 | 0.0 |
| | desviación | 0.2 | 0.3 | 2.3 | 0.5 |

![Desviación por episodio](reports/eda_episodios.png)

**Conclusiones:**
- **sobrecalentamiento:** la temperatura media sube a 88 °C (vs 55 en normal) y el reloj baja a ~2100 MHz por throttling.
- **degradacion_memoria:** se parece a normal en todo excepto en los **errores ECC**: ~5 por segundo contra ~0.
- **falla_alimentacion:** se delata por la **desviación**: el reloj varía ~204 MHz contra ~30 en normal, y la potencia ~45 W contra ~20.
- Media y mediana casi coinciden en todas las señales: no hay valores extremos que distorsionen la media.

## Features: el CSV de entrenamiento

Cada episodio se parte en **10 ventanas de 30 segundos**, sin mezclar episodios, así que cada episodio aparece 10 veces en `data/features_train.csv` (480 filas). De cada ventana se calculan **media, mediana, desviación, mínimo y máximo** de las 5 señales: 25 features.

Se entrena con ventanas y no con una fila por episodio porque la API recibe ventanas, porque así hay 480 ejemplos en vez de 48, y porque el enunciado lo pide.

`estado`, `episodio_id`, `ventana` y `segundo` **no** son features. `data.py` (entrenamiento) y `predict.py` (API) usan la misma función `calcular_features()` de `src/features.py`.

## Prueba de modelos

Se compararon 4 modelos con **GroupKFold por episodio** (5 particiones): las ventanas de un mismo episodio nunca quedan repartidas entre train y test, así no hay fuga.

| Modelo | Accuracy | Desviación |
|---|---|---|
| Regresión logística | 1.000 | 0.000 |
| Árbol de decisión | 1.000 | 0.000 |
| Random Forest | 1.000 | 0.000 |
| Gradient Boosting | 0.994 | 0.008 |

![Matrices de confusión](reports/matrices_confusion.png)

**Modelo final: regresión logística.** Empata en 100% con el árbol y Random Forest, sin errores en la matriz de confusión. Es el más simple de los tres, es rápido, y sus coeficientes muestran qué feature empuja hacia cada estado. Va con `StandardScaler` en el mismo pipeline porque la regresión logística es sensible a la escala (el reloj está en miles y los ECC entre 0 y 5).

## Validación

Las mismas reglas físicas se aplican al entrenar y al predecir, con pandera en los dos lados:

| | Entrenamiento | API |
|---|---|---|
| Qué valida | Telemetría cruda para entrenar | Ventana que llega a predecir |
| Estructura | Pandera: las 8 columnas, ni extra ni faltantes | BaseModel: los 5 campos, tipos correctos, mínimo 10 lecturas |
| Valores | Pandera (`esquema`): temp 0–110 °C, potencia 0–400 W, uso 0–100 %, reloj 0–3000 MHz, ECC ≥ 0, sin nulos, estado válido | Pandera (`esquema_lecturas`): **los mismos rangos físicos** |
| Si falla | Descarta la fila; si pasan del 1 %, se detiene | Quita las lecturas inválidas; si quedan menos de 10 válidas, responde **422** sin tocar el modelo |

Ejemplos de cómo responde la API:

| Qué llega | Respuesta |
|---|---|
| 30 lecturas válidas | 200 con la predicción |
| 30 lecturas, una con potencia −13.5 | 200: se quita esa lectura y se predice con las 29 restantes |
| Ventana con la temperatura en Fahrenheit | 422: *"Solo 0 de 30 lecturas tienen valores validos... temp_c en Celsius"* |
| 3 lecturas | 422: *"List should have at least 10 items"* |
| Falta un campo o viene texto | 422 del BaseModel, indicando la lectura y el campo |

- **Tests:** `tests/test_schema.py` comprueba que el contrato descarta nulos, potencias negativas, ECC negativos, estados inválidos y Fahrenheit, y que falla si falta o sobra una columna. `tests/test_api.py` comprueba las respuestas 200 y 422 de la API.
- **Fahrenheit:** una GPU a 55 °C son 131 °F, fuera del rango 0–110. Al entrenar, pandera descarta esas filas y el entrenamiento se detiene; en la API, la ventana responde 422.
- **Ventana de 3 lecturas:** se rechaza porque con tan pocos datos la desviación no es confiable, y la desviación es justo lo que delata la falla de alimentación