# Pasos para iniciar TensorBoard y Jupyter

## 1. Construir la imagen de docker

docker build -t tensorflowdemo:1.0 .

## 2. Crear un contenedor o mini-pc a partir de la imagen anterior

docker run --name tensorflowdemo -it tensorflowdemo:1.0 -p 6006:6006 -p 8886:8886 -p 8888:8888

## Iniciar Jupyter dentro del contenedor (Consola No 1)

1. Entrar al contenedor

docker exec -i -t tensorflowdemo bash

2. Iniciar jupyter notebook en el puerto 8888

jupyter notebook

## Iniciar tensorBoard dentro del contenedor (Consola No 2)

1. Entrar al contenedor

docker exec -i -t tensorflowdemo bash

2. Iniciar tensorBoard

tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006


## Acceder a Jupyter y tensorBoard

Verifica la ip generada por docker, luego abre un explorador (Chrome/Mozilla) y entra a la direccion ip:8888, podras comprobar que jupyter esta iniciado.   Para comprobar que tensorBoard esta iniciado, abre nueva pestaña y entra a la direccion ip:6006.
