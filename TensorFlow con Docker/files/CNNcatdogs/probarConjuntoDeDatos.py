import conjuntoDeDatos as data
import os
import glob
import numpy as np
import cv2

rutaDeDatos='F:\\umg media\\IA\\CNN\\imagenes\\catdogs\\train'
clases = ['dog','cat']
numeroClases = len(clases)
tamanoDeDataDeValidacion = 0.2
tamanoDeImagenes = 128

def probar():
    resultado = data.leerDatosDeEntrenamiento(rutaDeDatos, 
                tamanoDeImagenes, 
                clases, 
                tamanoDeDataDeValidacion)
    print(resultado)

probar()