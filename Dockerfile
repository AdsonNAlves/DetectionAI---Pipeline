FROM python:3.10-slim

# Instala dependências do sistema (inclui git e build essentials para futuras arquiteturas)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        vim \
        && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY ../requirements.txt /app/requirements.txt
COPY . /app/

# Instala dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

#ENV PYTHONPATH=/app/src
#--entrypoint /bin/bash
RUN chmod +x /app/training.sh
#CMD ["sh", "training.sh"]