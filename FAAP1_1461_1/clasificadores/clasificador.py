import numpy as np
from abc import ABCMeta, abstractmethod


class Clasificador(object):
    # Clase abstracta
    __metaclass__ = ABCMeta
    particiones = []
    errores = []

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
    # de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self, particionado, dataset, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test
        self.particiones = particionado.creaParticiones(dataset.datos)

        self.errores = []
        predicciones = []

        for particion in self.particiones:
            entrenamiento = dataset.extraeDatosTrain(particion.indicesTrain)
            _validacion = dataset.extraeDatosTest(particion.indicesTrain)
            y_test = _validacion[:, -1]
            x_test = _validacion[:, :-1]

            self.entrenamiento(entrenamiento, dataset.nominalAtributos, dataset.diccionarios)
            prediccion = self.clasifica(x_test, dataset.nominalAtributos, dataset.diccionarios)

            error = float(np.sum(prediccion != y_test)) / len(y_test)

            self.errores.append(error)

            predicciones.append(prediccion)

        return predicciones

##############################################################################
