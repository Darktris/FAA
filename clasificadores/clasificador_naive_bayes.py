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
    parametros_normales = []
    
    # Formula de Bayes: P(hip|datos) = P(datos|hip)*P(hip) = vero*priori
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        
        numero_atributos = len(diccionario) - 1
        numero_clases = len(diccionario[-1])

        # Inicializamos 
        self.lista_tablas_probabilidades = [None for _ in range(numero_atributos)]
        self.prioris = [None] * numero_clases
        self.parametros_normales = [{'mu': 0, 'sigma': 0}] * numero_atributos
        # Para cada clase
        for indice_clase, clase in enumerate(diccionario[-1].values()):

            # Casos totales = datos en los que la clase coincida
            indices_casos_totales = datostrain[:, -1] == clase
            # Numero de casos totales de esa clase
            cardinalidad_clase = float(indices_casos_totales.sum())
            
            # Priori: datos de esa clase / datos totales
            self.prioris[indice_clase] = (cardinalidad_clase) / (len(datostrain[:, -1]))

            # Para cada atributo
            for indice_atributo in range(numero_atributos):
                # Una tabla por atributo
                numero_valores_atributo = len(diccionario[indice_atributo].keys())
                # Diccionario que contiene para todo valor de ese atributo, un 0
                dictionary = [(value , 0) for value in diccionario[indice_atributo].values()]
                # La tabla contiene en ese atributo contiene el diccionario tantas veces como el numero de clases
                self.lista_tablas_probabilidades[indice_atributo] = [dict(dictionary)] * numero_clases
                # Si es discreto
                if atributosDiscretos[indice_atributo]:
                    # Para cada valor del atributo
                    for valor_atributo in diccionario[indice_atributo].values():
                        # indices casos favorables son los datos en los que el valor del atributo esta y la clase es la deseada
                        indices_casos_favorables = (datostrain[:, indice_atributo] == valor_atributo) & (datostrain[:, -1] == clase)
                        # Numero de casos favorables
                        numero_casos_favorables = float(indices_casos_favorables.sum())
                        # Verosimilitud es numero de casos favorables entre todos los de la clase
                        self.lista_tablas_probabilidades[indice_atributo][indice_clase][valor_atributo] = (numero_casos_favorables+self.laplace_smoothing) / (cardinalidad_clase + self.laplace_smoothing*numero_valores_atributo*numero_atributos)
                        #print(numero_casos_favorables, cardinalidad_clase)
                        #print('attr:', indice_atributo, 'clase:', indice_clase, 'valor_attr:',valor_atributo, self.lista_tablas_probabilidades[indice_atributo][indice_clase][valor_atributo])


                else:
                    # Calculo media 
                    self.parametros_normales[indice_atributo]['mu'] = np.mean(datostrain[:, indice_atributo])
                    # Calculo varianza
                    self.parametros_normales[indice_atributo]['var'] = np.var(datostrain[:, indice_atributo])


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
        
        # Inicializamos con las probabilidades a priori para cada clase
        probabilidades = np.asarray([[ self.prioris[indice_clase] for indice_clase in diccionario[-1].values() ]] * len(datostest))
        # Para cada registro del dataset de clasificacion
        for indice_fila, fila in enumerate(datostest):
            # Para cada atributo del registro
            for indice_atributo, valor_atributo in enumerate(fila[:-1]):
                # Para cada clase
                for indice_clase in diccionario[-1].values():
                    if atributosDiscretos[indice_atributo]:
                        probabilidades[indice_fila][indice_clase]*=self.lista_tablas_probabilidades[indice_atributo][indice_clase][valor_atributo]
                        #print('fila:', indice_fila, 'attr:', indice_atributo, 'clase:', indice_clase, 'valor_attr:',valor_atributo, self.lista_tablas_probabilidades[indice_atributo][indice_clase])
                    else:
                        mu = self.parametros_normales[indice_atributo]['mu']
                        sigma = self.parametros_normales[indice_atributo]['sigma']
                        if sigma != 0:
                            probabilidades[indice_fila][indice_clase]*=float(norm.pdf(valor_atributo, mu, sigma))
        prediccion = np.argmax(probabilidades, axis=1)
        #for probabilidad in probabilidades:
        #    print(probabilidad)
        
        for _i in range(len(prediccion)):
            print(prediccion[_i], probabilidades[_i])
        #print('argmax', np.argmax(probabilidades, axis=1), probabilidades)

        return prediccion
