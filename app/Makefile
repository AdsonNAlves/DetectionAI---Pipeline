# ==============================
# Configurações
# ==============================
CONTAINER_RUNTIME ?= podman
IMAGE_NAME = vision-api
CONTAINER_NAME = vision-api-container
PORT = 8080

# ==============================
# Targets principais
# ==============================
.PHONY: help build run run-gpu shell stop clean lint init

help:  ## Mostra esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build da imagem
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME) .

run: ## Rodar container sem GPU
	$(CONTAINER_RUNTIME) run -d --rm \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8080 \
		-v $(shell pwd):/app \
		$(IMAGE_NAME)

run-gpu: ## Rodar container com GPU (NVIDIA)
	$(CONTAINER_RUNTIME) run -d --rm \
		--name $(CONTAINER_NAME) \
		--security-opt=label=disable \
		--hooks-dir=/usr/share/containers/oci/hooks.d \
		--device nvidia.com/gpu=all \
		-p $(PORT):8080 \
		$(IMAGE_NAME)

shell: ## Acessar o shell do container
	$(CONTAINER_RUNTIME) exec -it $(CONTAINER_NAME) /bin/bash

stop: ## Parar container
	$(CONTAINER_RUNTIME) stop $(CONTAINER_NAME)

clean: ## Remover imagem e containers parados
	-$(CONTAINER_RUNTIME) rm -f $(CONTAINER_NAME) || true
	-$(CONTAINER_RUNTIME) rmi -f $(IMAGE_NAME) || true

lint: ## Checar estilo de código Python
	@echo ">> Rodando flake8..."
	flake8 .

init: ## Instalar dependências locais
	@echo ">> Instalando dependências..."
	pip install -r requirements.txt
