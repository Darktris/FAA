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
        col_clases = np.asarray(datostrain[:, numero_atributos-1])
        self.priori = np.asarray(self.crearHistograma(col_clases))/len(col_clases)
        self.clases = np.asarray((diccionario[numero_atributos-1].keys()))
        self.posteriori = (numero_atributos-1)*[None]
        self.posteriori_args = (numero_atributos-1)*[None]
        for i in range(numero_atributos-1):
            discreto = atributosDiscretos[i]
            columna = datostrain[:,i]
            columnaClasificada = zip(columna, col_clases)
            self.posteriori_args[i] = {}
            self.posteriori[i] = {}
            #print(self.clases)
            if discreto:
                hist = np.asarray(self.crearHistograma(columna))
                for i_clase in range(len(self.clases)):
                    clase = self.clases[i_clase]
                    self.posteriori_args[i][clase] = {}
                    #
                    hist = np.asarray(self.crearHistograma(columnaClasificada[col_clases == clase]))/len(columnaClasificada[col_clases == clase])
                    self.posteriori_args[i][clase]['hist'] = hist*self.priori[clase==self.clases]
                    from IPython.core.debugger import Tracer; Tracer()()
                    self.posteriori[i][clase] = self.posterioriDiscreto
            else:
                self.posteriori_args[i][clase] = {}
                self.posteriori_args[i][clase]['mu'] = np.mean(columna)
                self.posteriori_args[i][clase]['sigma'] = np.var(columna)
                self.posteriori_args[i][clase]['priori'] = self.priori[clase==self.clases]
                self.posteriori[i][clase] = self.posterioriContinuo
    
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
        return np.histogram(x, bins=len(np.unique(x)-1))[0]

    def clasifica(self, datostest, atributosDiscretos, diccionario):
        resultado = []
        for record in datostest:
            prob = {}
            for clase in self.clases:
                prob[clase] = 1.0
                for fieldIndex in range(len(record)):
                    prob[clase] *= self.posteriori[fieldIndex][clase](record[fieldIndex], self.posteriori_args[fieldIndex][clase])
                from IPython.core.debugger import Tracer; Tracer()()

            clase = np.argmax(prob)
            resultado.append(clase)

        return resultado
                
            
