FROM python:3.12-slim

WORKDIR /app

# primero las dependencias: si solo cambia el codigo, Docker reutiliza esta capa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# codigo y modelo YA entrenado (no se entrena dentro del contenedor)
COPY src/ src/
COPY models/modelo.joblib models/modelo.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]