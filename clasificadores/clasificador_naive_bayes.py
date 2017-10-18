from clasificadores.clasificador import Clasificador
import numpy  as np
from scipy.stats import norm

class ClasificadorNaiveBayes(Clasificador):
    laplace_smoothing = 0
    args = []
    posteriori_args = []
    posteriori = []
    clase = []
    # Formula de Bayes: P(hip|datos) = P(datos|hip)*P(hip) = vero*priori
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        numero_atributos = len(datostrain[0, :])
        col_clases = np.asarray(datostrain[:, -1])

        self.priori = np.asarray(self.crearHistograma(col_clases))
        self.clases = np.asarray(list(diccionario[-1].keys()))
        self.posteriori = [None]*(numero_atributos-1)
        self.posteriori_args = [None]*(numero_atributos-1)

        # from IPython.core.debugger import Tracer;
        # Tracer()()

        for indice in range(numero_atributos-1):
            discreto = atributosDiscretos[indice]
            columna = datostrain[:,indice]

            self.posteriori_args[indice] = {}
            self.posteriori[indice] = {}

            #print(self.clases)
            if discreto:
                for i_clase in range(len(self.clases)):
                    clase = self.clases[i_clase]
                    self.posteriori_args[indice][clase] = {}

                    attribute_filtered = [elem for index, elem in enumerate(columna) if (col_clases[index] == clase)]
                    hist = np.asarray(self.crearHistograma(attribute_filtered))

                    self.posteriori_args[indice][clase]['hist'] = self.priori[i_clase]*hist

                    # from IPython.core.debugger import Tracer; Tracer()()
                    self.posteriori[indice][clase] = self.posterioriDiscreto
            else:
                self.posteriori_args[indice][clase] = {}
                self.posteriori_args[indice][clase]['mu'] = np.mean(columna)
                self.posteriori_args[indice][clase]['sigma'] = np.var(columna)
                self.posteriori_args[indice][clase]['priori'] = self.priori[clase==self.clases]
                self.posteriori[indice][clase] = self.posterioriContinuo
    
    def posterioriDiscreto(self, x, argumentos):
        hist = argumentos['hist']
        index = int(x)
        if index in hist:
            return float(hist[int(x)])
        else:
            return 0
    
    def posterioriContinuo(self, x, argumentos):
        mu = argumentos['mu']
        sigma = argumentos['sigma']
        priori = argumentos['priori']
        return float(norm.pdf(x, mu, sigma)*priori)
    
    def crearHistograma(self, x):

        hist = np.ndarray(shape=(len(x),1))
        bins = sorted(np.unique(x))
        total = len(x)

        for index, bin in enumerate(bins):
            hist[index] = (x == bin).sum() / total

        return hist


    def clasifica(self, datostest, atributosDiscretos, diccionario):
        resultado = []
        for record in datostest:
            prob = {}
            for clase in self.clases:
                prob[clase] = 1.0
                for fieldIndex in range(len(record)):
                    prob[clase] *= self.posteriori[fieldIndex][clase](record[fieldIndex], self.posteriori_args[fieldIndex][clase])


            clase = np.argmax(prob)
            resultado.append(clase)

        return resultado
                
            
