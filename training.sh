#!/bin/bash

last="None"
data=20250915
version=v1
training=True #True|False
arch="v11" #v8
weights="caminho/do/seu/checkpoint.pt"

experiment_name=${data}_version_${version}_last_${last}

## SAVING MODEL - default = experiment_name
SAVED_PROJECT="checkpoint/"

python3 main.py --project=${SAVED_PROJECT} --name=${experiment_name} --training=${training} --arch=${arch} --weights=${weights}