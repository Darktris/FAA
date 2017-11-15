# import multiprocessing
import time

import numpy as np
from numpy.linalg import norm

from clasificadores.clasificador import Clasificador


class __C__(object):
    def __init__(self, datoTest):
        self.datoTest = datoTest

    def __call__(self, src):
        return __distancia__(src, self.datoTest)


class ClasificadorVecinosProximos(Clasificador):
    probabilidades = []

    datos_train = None
    datos_normalizados_train = None
    clases_train = None
    atributos_continuos = None

    def __init__(self, K=5):
        self.K = K

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        self.atributos_continuos = np.logical_not(atributosDiscretos)
        self.datos_train = datosTrain[:, :-1]
        self.datos_normalizados_train = self.normalizarDatos(datosTrain[:, :-1])

        self.clases_train = datosTrain[:, -1]

        return

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        return np.fromiter(map(lambda record: self.__clasifica_uno__(record), datosTest), dtype=float)

    def __clasifica_uno__(self, datoTest):
        start = time.clock()

        # pool = multiprocessing.Pool(processes=4)

        # Fancy optimization :(
        # distancias = pool.map(__C__(datoTest), self.datos_normalizados_train)
        distancias = np.fromiter(map(lambda ejemplo: __distancia__(ejemplo, datoTest), self.datos_normalizados_train),
                                 dtype=float)

        step_1 = time.clock()

        indices_vecinos = np.argpartition(distancias, kth=self.K)[:self.K]

        clases_vecinos = self.clases_train[indices_vecinos]
        step_2 = time.clock()

        unique, counts = np.unique(clases_vecinos, return_counts=True)
        step_3 = time.clock()

        return unique[np.argmax(counts)]

    def calcularMedias(self, datostrain):

        medias = np.zeros(datostrain.shape[1])
        desv_est = np.ones(datostrain.shape[1])

        for indice, columna_continuo in enumerate(zip(datostrain.T, self.atributos_continuos)):
            columna, continuo = columna_continuo
            if continuo:
                medias[indice] = np.mean(columna)
                desv_est[indice] = np.std(columna)

        return medias, desv_est

    def normalizarDatos(self, datos):
        medias, desv_est = self.calcularMedias(datos)
        return (datos - medias) / desv_est


def __distancia__(vector_1, vector_2):
    return norm(vector_1 - vector_2, ord=2)
