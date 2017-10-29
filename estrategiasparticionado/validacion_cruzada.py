import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from estrategiasparticionado.estrategias_particionado import EstrategiaParticionado
from estrategiasparticionado.particion import Particion
import numpy as np
import random
from functools import reduce


class ValidacionCruzada(EstrategiaParticionado):
    nfolds = 10
    nombreEstrategia = "ValidacionCruzada"

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones
    # y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)

        # Calculo del numero de indices no alineados
        numero_no_alineados = len(datos) % self.nfolds

        # Generacion de la permutacion de los indices
        indices = np.random.permutation(len(datos))

        # Separacion de los indices alineados y no alineados
        indices_alineados = indices[:len(datos) - numero_no_alineados]
        indices_no_alineados = indices[len(datos) - numero_no_alineados:]

        # Hacemos una particion de los alineados en arrays del tamanno deseado
        indices_particiones_alineados = np.reshape(indices_alineados, (self.nfolds, -1))

        # inicializamos la lista de las particiones finales con las que inicializar
        indices_particiones = [None] * self.nfolds

        # Distribuimos los indices no alineados incluyendolos al final de los primeros alineados
        for i in range(numero_no_alineados):
            indices_particiones[i] = np.append(indices_particiones_alineados[i], indices_no_alineados[i])

        # cuando hemos terminado de distribuir los no alineados, insertamos los que no necesitan
        # que se les annada ningun indice al final, tendran un elemento menos estas ultimas particiones
        for i in range(numero_no_alineados, self.nfolds):
            indices_particiones[i] = indices_particiones_alineados[i]

        # Inicializacion de las particiones a devolver
        self.particiones = [Particion() for _ in range(self.nfolds)]

        for particion, index in zip(self.particiones, range(self.nfolds)):
            particion.indicesTest = indices_particiones[index]
            
            def pegar(x,y):
                return np.append(x,y)
            
            if index > 0 and index < len(self.particiones) -1:
                indices_izq = reduce(pegar,indices_particiones[:index])
                indices_dcha = reduce(pegar,indices_particiones[index+1:])
                particion.indicesTrain = np.append(indices_izq,indices_dcha)
            elif index ==0:
                particion.indicesTrain = reduce(pegar,indices_particiones[1:])
            else:
                particion.indicesTrain = reduce(pegar,indices_particiones[:-1])
                
            

        self.numeroParticiones = self.nfolds

        return self.particiones
