import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from EstrategiasParticionado.EstrategiaParticionado import EstrategiaParticionado
from EstrategiasParticionado.Particion import Particion
import numpy as np
import random


class ValidacionCruzada(EstrategiaParticionado):
    nfolds = 10
    nombreEstrategia = "ValidacionCruzada"

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones
    # y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)

        numero_no_alineados = len(datos) % self.nfolds

        indices = np.random.permutation(len(datos))

        indices_alineados = indices[:len(datos) - numero_no_alineados ]
        indices_no_alineados = indices[len(datos) - numero_no_alineados+1:]

        indicesParticiones = np.reshape(indices_alineados, (self.nfolds, -1))

        print(indicesParticiones)

        for i in range(numero_no_alineados):
            np.append( indicesParticiones[i], indices_no_alineados[i])

        self.particiones = [Particion() for _ in range(self.nfolds)]

        for particion, index in zip(self.particiones, range(self.nfolds)):
            particion.indicesTrain = indicesParticiones[index]
            particion.indicesTest = indicesParticiones[:index] + indicesParticiones[index + 1:]


        self.numeroParticiones = self.nfolds

        return self.particiones
