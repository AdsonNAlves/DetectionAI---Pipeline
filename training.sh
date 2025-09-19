#!/bin/bash

last="None"
data=20250919
version=v1
training=True #True|False
arch="frcnn" #use: 'v8', 'v11' or 'frcnn'
weights="caminho/do/seu/checkpoint.pt"

experiment_name=${data}_version_${version}_last_${last}_arch_${arch}

## SAVING MODEL - default = experiment_name
SAVED_PROJECT="checkpoint/"

python3 main.py --project=${SAVED_PROJECT} --name=${experiment_name} --training=${training} --arch=${arch} --weights=${weights}
