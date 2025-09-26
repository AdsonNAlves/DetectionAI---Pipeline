FROM python:3.10-slim

# Evita cache de .pyc e melhora logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cria pasta da aplicação
WORKDIR /app

# Instala dependências de sistema necessárias para OpenCV, Torch etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copia dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Porta da API
EXPOSE 8080

# Comando para subir o server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# WORKDIR /app
# COPY requirements.txt .
# RUN apt-get update && apt-get install -y python3-pip
# RUN pip install -r requirements.txt

# COPY app/ ./app/

# EXPOSE 8080
# CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]