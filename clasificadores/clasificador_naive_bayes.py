from clasificadores.clasificador import Clasificador


class ClasificadorNaiveBayes(Clasificador):
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        numero_atributos = len(datostrain[0, :]) - 1

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        pass
