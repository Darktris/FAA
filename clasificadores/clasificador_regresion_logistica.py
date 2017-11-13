import numpy as np

from clasificadores.clasificador import Clasificador


class ClasificadorRegresionLogistica(Clasificador):
    tasa_aprendizaje = 0.01
    n_epocas = 100
    w = []

    probabilidades = []
    prediccion = []

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        numero_atributos = len(diccionario) - 1

        # Inicializamos
        self.w = np.random.uniform(low=-1.0, high=1.0, size=numero_atributos + 1)
        # Para cada epoca
        for epoca in range(self.n_epocas):
            # Para cada muestra
            for indice_fila, fila in enumerate(datostrain):
                # Calculamos el x real
                current = np.insert(fila[:-1], 0, 1.0, axis=0)
                # Calculamos la sigmoidal
                sigma = self.sigmoidal(fila[:-1])
                # Actualizamos el valor de w
                self.w = self.w - (self.tasa_aprendizaje * (sigma - fila[-1]) * current)

    def sigmoidal(self, sample):
        x_real = np.insert(sample, 0, 1.0, axis=0)
        prod_escalar = np.sum(x_real * self.w)
        return 1 / (1 + np.exp(-prod_escalar))

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        self.probabilidades = np.zeros(len(datostest))
        # Para cada registro del dataset de clasificacion
        for indice_fila, fila in enumerate(datostest):
            self.probabilidades[indice_fila] = self.sigmoidal(fila)

        self.prediccion = np.zeros(len(datostest))
        self.prediccion[self.probabilidades >= 0.5] = 1.0;

        return self.prediccion
