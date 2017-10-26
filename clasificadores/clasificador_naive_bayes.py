import numpy  as np
from scipy.stats import norm

from clasificadores.clasificador import Clasificador


class ClasificadorNaiveBayes(Clasificador):
    laplace_smoothing = 0.0
    clases = []

    probabilidades = []
    prioris = []
    parametros_normales = []
    factores = []

    # Formula de Bayes: P(hip|datos) = P(datos|hip)*P(hip) = vero*priori
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        numero_atributos = len(diccionario) - 1
        tamano_train = len(datostrain[:, -1])
        self.clases = diccionario[-1].values()

        # Inicializamos
        self.prioris = {}
        self.cardinalidades_clase = {}

        self.probabilidades = [None] * numero_atributos
        self.factores = [None] * numero_atributos
        self.parametros_normales = [{'mu': 0, 'sigma': 0}] * numero_atributos

        # Rellenamos los datos relacionados con las clases en el entrenamiento
        for clase in self.clases:
            # Casos totales = numero de datos en los que la clase coincida
            indices_casos_totales = (datostrain[:, -1] == clase)
            # Numero de casos totales de esa clase
            self.cardinalidades_clase[clase] = float(np.sum(indices_casos_totales))

            # Priori: datos de esa clase / datos totales
            self.prioris[clase] = self.cardinalidades_clase[clase] / tamano_train

        # Para cada atributo
        for indice_atributo in range(numero_atributos):
            # Un diccionario por atributo
            valores_atributo = diccionario[indice_atributo].values()
            numero_valores_atributo = len(valores_atributo)
            # Diccionario que contiene para to_do valor de ese atributo, un 0
            # La tabla contiene en ese atributo contiene el diccionario tantas veces como el numero de clases
            self.probabilidades[indice_atributo] = {}
            self.factores[indice_atributo] = {}

            for valor_atributo in valores_atributo:
                self.probabilidades[indice_atributo][valor_atributo] = {}

                self.factores[indice_atributo][valor_atributo] = (datostrain[:,
                                                                  indice_atributo] == valor_atributo).sum() / len(
                    datostrain[:, indice_atributo])

                # Si es discreto
                if atributosDiscretos[indice_atributo]:
                    # Para cada valor del atributo
                    for clase in self.clases:
                        # indices casos favorables son los datos en los que el valor del atributo esta y la clase es la deseada
                        indices_casos_favorables = (datostrain[:, indice_atributo] == valor_atributo) & (
                            datostrain[:, -1] == clase)
                        # Numero de casos favorables
                        numero_casos_favorables = float(np.sum(indices_casos_favorables))
                        # Verosimilitud es numero de casos favorables entre todos los de la clase
                        numerador = numero_casos_favorables + self.laplace_smoothing
                        denominador = (self.cardinalidades_clase[clase] + self.laplace_smoothing
                                       * numero_valores_atributo * numero_atributos)
                        probabilidad = numerador / denominador

                        self.probabilidades[indice_atributo][valor_atributo][clase] = probabilidad

                else:
                    # Calculo media
                    self.parametros_normales[indice_atributo]['mu'] = np.mean(datostrain[:, indice_atributo])
                    # Calculo varianza
                    self.parametros_normales[indice_atributo]['var'] = np.var(datostrain[:, indice_atributo])
                    # print(self.probabilidades[indice_atributo])

        self.experiment = ((datostrain[:, 0] == 2) & (datostrain[:, -1] == 1)).sum()
        self.experiment_2 = (datostrain[:, -1] == 1).sum()

    def posterioriContinuo(self, x, argumentos):
        mu = argumentos['mu']
        sigma = argumentos['sigma']
        priori = argumentos['priori']
        return float(norm.pdf(x, mu, sigma) * priori)

    def print_probabilidades(self, diccionario):

        print("Numero atributos:")
        print("Priroris:")
        for clase in self.clases:
            print("\tClase:", clase, "prob:", self.prioris[clase])

        for index, tabla in enumerate(self.probabilidades):
            print("Atributo:", index)
            print(diccionario[index])
            for value, fila in tabla.items():
                print("\tValor atributo:", value, "priori:", self.factores[index][value])
                for clase, prob in fila.items():
                    print("\t\tClase:", clase, "prob", prob)
        print("\n")

    def clasifica(self, datostest, atributosDiscretos, diccionario):
        # print(len(self.probabilidades))
        # for tabla in self.probabilidades:
        #     print(tabla)

        # Inicializamos con las probabilidades a priori para cada clase
        self.print_probabilidades(diccionario)

        probabilidades = [dict(self.prioris) for _ in datostest[:, -1]]
        probabilidades[0][0] = 1

        # Para cada registro del dataset de clasificacion
        for indice_fila, fila in enumerate(datostest):
            # Para cada atributo del registro
            # print("Fila:",fila)
            for indice_atributo, valor_atributo in enumerate(fila[:-1]):
                # Para cada clase
                for clase in self.clases:
                    # if indice_fila == 0:
                    # print("Attr:", indice_atributo, "Valor atrib:", valor_atributo, "clase:", clase, "prob:",
                    #      probabilidades[indice_atributo][clase])
                    # print("Post:", self.probabilidades[indice_atributo][valor_atributo][clase])
                    # print("Prob c1:", probabilidades[indice_fila][0])
                    # print("Prob c2:", probabilidades[indice_fila][1])
                    if atributosDiscretos[indice_atributo]:
                        #print(fila)
                        #print("Clase:",clase,self.prioris[clase])
                        #print("Atributo:",indice_atributo,"valor",valor_atributo)
                        #print("Prob:",self.probabilidades[indice_atributo][valor_atributo][clase])
                        probabilidades[indice_fila][clase] *= (self.probabilidades[indice_atributo][valor_atributo][
                                                                   clase] / self.factores[indice_atributo][
                                                                   valor_atributo])
                        #print("Actual:",probabilidades[indice_fila][clase])
                    else:
                        mu = self.parametros_normales[indice_atributo]['mu']
                        sigma = self.parametros_normales[indice_atributo]['sigma']
                        if sigma != 0:
                            probabilidades[indice_fila][clase] *= float(norm.pdf(valor_atributo, mu, sigma))
                            # print(probabilidades[indice_fila])

        # print("END")

        comparativa = []
        for probabilidad in probabilidades:
            fila = []
            for clase in self.clases:
                fila.append(probabilidad[clase])
            #print(fila)
            comparativa.append(fila)
        print(comparativa)
        prediccion = np.argmax(comparativa, axis=1)

        return prediccion
