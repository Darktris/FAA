import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from EstrategiasParticionado.EstrategiaParticionado import EstrategiaParticionado
from EstrategiasParticionado.Particion import Particion
import numpy as np
import random


class ValidacionSimple(EstrategiaParticionado):
    porcentajeEntrenamiento = 0.7
    nombreEstrategia = "ValidacionSimple"

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)

        indices = np.random.permutation(len(datos))

        indicesEntrenamiento = indices[:int(self.porcentajeEntrenamiento * len(datos))]
        indicesTest = indices[int(self.porcentajeEntrenamiento * len(datos)):]

        particion = Particion()

        particion.indicesTrain = indicesEntrenamiento
        particion.indicesTest = indicesTest
        self.numeroParticiones = 1
        self.particiones = [particion]

        return self.particiones
