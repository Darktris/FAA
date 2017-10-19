from clasificadores.clasificador import Clasificador
import numpy  as np
from scipy.stats import norm


class ClasificadorNaiveBayes(Clasificador):
    laplace_smoothing = 0
    args = []
    posteriori_args = []
    posteriori = []
    clase = []

    lista_tablas_probabilidades = []
    prioris = []

    # Formula de Bayes: P(hip|datos) = P(datos|hip)*P(hip) = vero*priori
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        numero_atributos = len(diccionario) - 1
        numero_clases = len(diccionario[-1])

        self.lista_tablas_probabilidades = [None for _ in range(numero_atributos)]
        self.prioris = [None] * numero_clases

        for indice_clase, clase in enumerate(diccionario[-1].values()):

            indices_casos_totales = datostrain[:, -1] == clase
            cardinalidad_clase = indices_casos_totales.sum()
            self.prioris[indice_clase] = cardinalidad_clase / len(datostrain[:, -1])

            for indice_atributo in range(numero_atributos):
                # Una tabla por atributo
                numero_valores_atributo = len(diccionario[indice_atributo].keys())
                dictionary = [(value , 0) for value in diccionario[indice_atributo].values()]
                self.lista_tablas_probabilidades[indice_atributo] = [dict(dictionary)] * numero_clases

                if atributosDiscretos[indice_atributo]:
                    # Caso discreto
                    for valor_atributo in diccionario[indice_atributo].values():
                        indices_casos_favorables = (datostrain[:, indice_atributo] == valor_atributo) & (datostrain[:, -1] == clase)
                        numero_casos_favorables = indices_casos_favorables.sum()

                        self.lista_tablas_probabilidades[indice_atributo][indice_clase][valor_atributo] = numero_casos_favorables / cardinalidad_clase

                else:
                    # TODO: Caso continuo
                    pass


    def posterioriContinuo(self, x, argumentos):
        mu = argumentos['mu']
        sigma = argumentos['sigma']
        priori = argumentos['priori']
        return float(norm.pdf(x, mu, sigma) * priori)


    def clasifica(self, datostest, atributosDiscretos, diccionario):
        #TODO: Terminar

        # print(len(self.lista_tablas_probabilidades))
        # for tabla in self.lista_tablas_probabilidades:
        #     print(tabla)

        probabilidades = np.asarray([[ self.prioris[indice_clase] for indice_clase in diccionario[-1].values() ]] * len(datostest))


        for indice_fila, fila in enumerate(datostest):
            for indice_atributo, valor_atributo in enumerate(fila[:-1]):
                for indice_clase in diccionario[-1].values():
                    probabilidades[indice_fila][indice_clase]*=self.lista_tablas_probabilidades[indice_atributo][indice_clase][valor_atributo]


        for probabilidad in probabilidades:
            print(probabilidad)

        print(prediccion)

        return prediccion
