# install PIP https://www.youtube.com/watch?v=zPMr0lEMqpo
# pip install opencv-python
# pip install -U scikit-learn
# python -m pip install --user numpy scipy

import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import ConjuntoDeImagenes 

def obtenerImagenDesdeRuta(rutaDeImagen,tamanoDeImagenes):
  imagen = cv2.imread(rutaDeImagen)
  imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
  imagen = cv2.resize(imagen, (tamanoDeImagenes, tamanoDeImagenes),0,0, cv2.INTER_LINEAR)
  imagen = imagen.astype(np.float64)
  imagen = np.multiply(imagen, 1.0 / 255.0)
  return imagen

def cargarDatosDeEntrenamiento(rutaDeDatos, tamanoDeImagenes, clases):
  imagenes = []
  etiquetas = []
  nombreDeImagenes = []
  grupoDeImagenes = []
  for clase in clases:
    indiceDeClase = clases.index(clase)
    #rutaDeDatosDeEntrenamiento = os.path.join(rutaDeDatos, clase, '*g')
    rutaDeDatosDeEntrenamiento=os.path.join(rutaDeDatos,clase+'.*.jpg')
    listaDeArchivos = glob.glob(rutaDeDatosDeEntrenamiento)

    index=0
    totalImages = len(listaDeArchivos)
    maxImages =  totalImages
    print('clase {}, indice {}, path {}, files number {}, max images {}'.format(clase, indiceDeClase, rutaDeDatosDeEntrenamiento,len(listaDeArchivos),maxImages))
    for archivo in listaDeArchivos:
      try:
        imagen = obtenerImagenDesdeRuta(archivo,tamanoDeImagenes)
        imagenes.append(imagen)
        etiqueta = np.zeros(len(clases))
        etiqueta[indiceDeClase] = 1.0
        etiquetas.append(etiqueta)

        nombreDeImagen = os.path.basename(archivo)
        nombreDeImagenes.append(nombreDeImagen)
        grupoDeImagenes.append(clase)
        
        if((index+1) % 1000 == 0):
          print('progress, read {} of {}'.format(index,totalImages))
        index = index + 1
      except Exception:
        pass

  imagenes = np.array(imagenes)
  etiquetas = np.array(etiquetas)
  nombreDeImagenes = np.array(nombreDeImagenes)
  grupoDeImagenes = np.array(grupoDeImagenes)

  print('End to get images')
  return imagenes, etiquetas, nombreDeImagenes, grupoDeImagenes


def leerDatosDeEntrenamiento(rutaDeDatosDeEntrenamiento, tamanoDeImagenes, clases, tamanoDeDataDeValidacion):
  class DataSets(object):
    pass
  resultadoDeImagenes = DataSets()

  imagenes, etiquetasDeImagenes, nombreDeImagenes, grupoDeImagenes = cargarDatosDeEntrenamiento(rutaDeDatosDeEntrenamiento, tamanoDeImagenes, clases)
  imagenes, etiquetasDeImagenes, nombreDeImagenes, grupoDeImagenes = shuffle(imagenes, etiquetasDeImagenes, nombreDeImagenes, grupoDeImagenes)

  if isinstance(tamanoDeDataDeValidacion, float):
    tamanoDeDataDeValidacion = int(tamanoDeDataDeValidacion * imagenes.shape[0])

  validacion_imagenes = imagenes[:tamanoDeDataDeValidacion]
  validacion_etiquetas = etiquetasDeImagenes[:tamanoDeDataDeValidacion]
  validacion_nombres = nombreDeImagenes[:tamanoDeDataDeValidacion]
  validacion_grupos = grupoDeImagenes[:tamanoDeDataDeValidacion]

  entrenamiento_imagenes = imagenes[tamanoDeDataDeValidacion:]
  entrenamiento_etiquetas = etiquetasDeImagenes[tamanoDeDataDeValidacion:]
  entrenamiento_nombres = nombreDeImagenes[tamanoDeDataDeValidacion:]
  entrenamiento_grupos = grupoDeImagenes[tamanoDeDataDeValidacion:]

  resultadoDeImagenes.entrenamiento = ConjuntoDeImagenes.ConjuntoDeImagenes(entrenamiento_imagenes, entrenamiento_etiquetas, entrenamiento_nombres, entrenamiento_grupos,'entrenamiento')
  resultadoDeImagenes.validacion = ConjuntoDeImagenes.ConjuntoDeImagenes(validacion_imagenes, validacion_etiquetas, validacion_nombres, validacion_grupos,'validacion')

  return resultadoDeImagenes
