# DetectionIA Pipeline

Este README explica como instalar Podman ou Docker, como usar a automação com o `Makefile` do projeto 

## Instalação do Podman ou Docker

Este projeto pode ser executado em containers usando **Podman** ou **Docker**.  
Escolha uma das opções abaixo para instalar:

### Podman

macOS (Homebrew):

```bash
brew install podman
podman machine init
podman machine start
```

Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y podman
```

Verifique:

```bash
podman --version
podman info
```

### Docker

macOS (Docker Desktop):

```bash
brew install --cask docker
# Abra e inicie o Docker Desktop
```

Debian/Ubuntu (resumido):

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmour -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
# Efetue logout/login para ativar o grupo docker
```

Verifique:

```bash
docker --version
docker info
```

> Atenção: em servidores é comum usar Podman (rootless) ou Docker com permissões controladas. Escolha conforme a política de segurança do ambiente.

---

## Automação com Make

O repositório inclui um `Makefile` para tarefas como construir imagens (`create-image`) e container (`create-container`). Comandos:

```bash
make create-image     # constrói a imagem usando o runtime configurado (docker/podman)
make create-container # controi o container para treinamento 
# dentro do container
./training.sh
```

## Ajustes rápidos no `Dockerfile`

Se você estiver construindo em um servidor com GPU e encontra erros como "Package 'libgl1-mesa-glx' has no installation candidate", remova esse pacote do `Dockerfile` e substitua por 'libgl1'.

Exemplo alterado:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
	 git \
	 build-essential \
	 libglib2.0-0 \
     libgl1 \
	 vim \
	 && rm -rf /var/lib/apt/lists/*
```
## Estrutura do diretório

Raiz do projeto:
.
├── common
│   └── utils.py
├── data
│   ├── train
│   │   ├── images
│   │   │   ├── 2025-04-24-08-40-18_315.jpg
│   │   │   ├── 2025-04-24-08-40-18_315.npy
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
├── networks
│   └── structures.py
├── notebooks
│   └── figures
│   └── results.ipynb
├── test
│   └── 20250915v1
│       ├── data.yaml
│       └── args.yaml
├── training.sh
├── Dockerfile
├── Makefile
├── main.py
├── test.sh
├── requirements.txt
├── yolo11m.pt
└── README.md

