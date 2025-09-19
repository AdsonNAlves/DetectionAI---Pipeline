.PHONY: help lint clean init

#################################################################################
# GLOBALS                                                                       #
#################################################################################
export DOCKER=podman
export BASE_IMAGE_NAME=iasense-detection
export BASE_DOCKERFILE=Dockerfile
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export CONTAINER_NAME=iasense-detection-container
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build docker image #### --build-arg UID=$(shell id -u)
create-image:
	$(DOCKER) build -t $(BASE_IMAGE_NAME) -f $(BASE_DOCKERFILE) --force-rm . 

## Run docker container
create-container: 
	$(DOCKER) run -it --entrypoint /bin/bash --group-add keep-groups -e NVIDIA_VISIBLE_DEVICES=GPU-3e98ebd8-0099-82a3-0389-bcd7ea84fc55 --shm-size=2g -v $(shell pwd):/app -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) --name $(CONTAINER_NAME) $(BASE_IMAGE_NAME)
