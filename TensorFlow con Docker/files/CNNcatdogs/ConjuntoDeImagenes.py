import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

class ConjuntoDeImagenes(object):
  """ contructor de la clase """
  def __init__(self, imagenes, etiquetas, nombres, clases):
    self._recuento = imagenes.shape[0]
    self._imagenes = imagenes
    self._etiquetas = etiquetas
    self._nombres = nombres
    self._clases = clases
    self._epochs_done = 0
    self._index_in_epoch = 0

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
  def epochs_done(self):
    return self._epochs_done

  def siguienteLote(self, tamanoDeLote):

    inicioDelLote = self._index_in_epoch
    self._index_in_epoch += tamanoDeLote

    if self._index_in_epoch > self._recuento:
      # After each epoch we update this
      self._epochs_done += 1
      inicioDelLote = 0
      self._index_in_epoch = tamanoDeLote
      assert tamanoDeLote <= self._recuento

    findDelLote = self._index_in_epoch

    print("siguiente lote, inicio: "+str(inicioDelLote)+", fin: "+str(finDelLote))

    return  self._imagenes[inicioDelLote:findDelLote], 
            self._etiquetas[inicioDelLote:findDelLote], 
            self._nombres[inicioDelLote:findDelLote], 
            self._clases[inicioDelLote:findDelLote]