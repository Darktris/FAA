import numpy as np
from numpy.linalg import norm

from clasificadores.clasificador import Clasificador

__ord__ = 2


class ClasificadorVecinosProximos(Clasificador):
    probabilidades = []

    datos_train = None
    datos_normalizados_train = None
    clases_train = None
    atributos_continuos = None

    def __init__(self, K=5, normalizar=True):
        self.K = K
        self.normalizar = normalizar

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        self.atributos_continuos = np.logical_not(atributosDiscretos)
        self.datos_train = datosTrain[:, :-1]
        if self.normalizar:
            self.datos_normalizados_train = self.normalizarDatos(datosTrain[:, :-1])
        else:
            self.datos_normalizados_train = datosTrain[:, :-1]

        self.clases_train = datosTrain[:, -1]

        return

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        if self.normalizar:
            datosTest = self.normalizarDatos(datosTest)
        return np.fromiter(map(lambda record: self.__clasifica_uno__(record), datosTest), dtype=float)

    def __clasifica_uno__(self, datoTest):

        distancias = np.fromiter(map(lambda ejemplo: __distancia__(ejemplo, datoTest), self.datos_normalizados_train),
                                 dtype=float)

        indices_vecinos = np.argpartition(distancias, kth=self.K)[:self.K]

        clases_vecinos = self.clases_train[indices_vecinos]

        unique, counts = np.unique(clases_vecinos, return_counts=True)

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
    return norm(vector_1 - vector_2, ord=__ord__)
