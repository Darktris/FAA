from abc import ABCMeta, abstractmethod


class EstrategiaParticionado(object):
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
    nombreEstrategia = "null"
    numeroParticiones = 0
    particiones = []

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos, seed=None):
        pass
