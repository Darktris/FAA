import numpy as np
from clasificadores.clasificador import Clasificador
from collections import Counter


class ClasificadorVecinosProximos(Clasificador):
    K = 5
    p = 2

    probabilidades = []

    datos_normalizados_train = None
    clases_train = None
    atributos_continuos = None

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        self.atributos_continuos = np.logical_not(atributosDiscretos)
        self.datos_normalizados_train = self.normalizarDatos(datosTrain)
        self.clases_train = datosTrain[:, -1]

        return

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        return np.fromiter(map(lambda record: self.__clasifica_uno__(record),datosTest))


    def __clasifica_uno__(self, datoTest):
        distancias = np.fromiter(map(lambda ejemplo: self.__distancia__(ejemplo,datoTest), self.datos_normalizados_train))
        indices_vecinos = np.argmin(distancias)[:self.K]
        clases_vecinos = self.clases_train[indices_vecinos]

        return clases_vecinos.sort(key=Counter(clases_vecinos).get, reverse=True)[0]

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

    def __distancia__(self, vector_1, vector_2):
        return np.linalg.norm(vector_1 - vector_2, ord=self.p)
