import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

class ConjuntoDeImagenes(object):
  """ contructor de la clase """
  def __init__(self, imagenes, etiquetas, nombres, clases,nombre):
    self._nombre= nombre
    self._recuento = imagenes.shape[0]
    self._imagenes = imagenes
    self._etiquetas = etiquetas
    self._nombres = nombres
    self._clases = clases
    self._epocasHechas = 0
    self._indiceDeEpoca = 0

  @property
  def nombre(self):
    return self._nombre

  @property
  def imagenes(self):
    return self._imagenes

  @property
  def etiquetas(self):
    return self._etiquetas

  @property
  def nombres(self):
    return self._nombres

  @property
  def clases(self):
    return self._clases

  @property
  def recuento(self):
    return self._recuento

  @property
  def epocasHechas(self):
    return self._epocasHechas

  def siguienteLote(self, tamanoDeLote):

    inicioDelLote = self._indiceDeEpoca
    self._indiceDeEpoca += tamanoDeLote

    if self._indiceDeEpoca > self._recuento:
      # After each epoch we update this
      self._epocasHechas += 1
      inicioDelLote = 0
      self._indiceDeEpoca = tamanoDeLote
      assert tamanoDeLote <= self._recuento

    finDelLote = self._indiceDeEpoca
    print("siguiente lote de "+str(self._nombre)+", inicio: "+str(inicioDelLote)+", fin: "+str(finDelLote))
    return  self._imagenes[inicioDelLote:finDelLote], self._etiquetas[inicioDelLote:finDelLote], self._nombres[inicioDelLote:finDelLote], self._clases[inicioDelLote:finDelLote]
