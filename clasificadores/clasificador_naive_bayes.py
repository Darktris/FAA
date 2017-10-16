from clasificadores.clasificador import Clasificador
import numpy  as np
from scipy.stats import norm

class ClasificadorNaiveBayes(Clasificador):
    laplace_smoothing = 0
    args = []
    
    # Formula de Bayes: P(hip|datos) = P(datos|hip)*P(hip) = vero*priori
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        numero_atributos = len(datostrain[0, :])
        col_clases = datostrain[:, numero_atributos-1]
        self.priori = np.hist(col_clases)/len(col_clases)
        self.clases = keys(diccionario[numero_atributos-1])
        
        for index, discreto in zip(range(numero_atributos-1), atributosDiscretos):
            columna = datostrain[:,index]
            columnaClasificada = zip(columna, col_clases)
            self.posteriori_args[i] = {}
            self.posteriori[i] = {}
            if discreto:
                hist = np.histogram(columna)
                for clase in self.clases:
                    self.posteriori_args[i][clase] = {}
                    self.posteriori_args[i][clase]['hist'] = np.hist(columnaClasificada[col_clases == clase])*self.priori[clase==self.clases]
                    self.posteriori[i][clase] = self.posterioriDiscreto
            else:
                self.posteriori_args[i]['mu'] = np.mean(columna)
                self.posteriori_args[i]['sigma'] = np.var(columna)
                self.posteriori_args[i]['priori'] = self.priori[clase==self.clases]
                self.posteriori[i][clase] = self.posterioriContinuo
    
    def posterioriDiscreto(x, args):
        hist = args['hist']
        return hist[x]
    
    def posterioriContinuo(x, args):
        mu = args['mu']
        sigma = args['sigma']
        priori = args['priori']
        return norm.pdf(x, mu, sigma)*priori
    
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        resultado = []
        for record in datostest:
            prob = []
            for clase in self.clases:
                prob[clase] = 1
                for fieldIndex in range(len(record)):
                    prob[clase] *= self.posteriori[fieldIndex](record[fieldIndex], self.posteriori_args[fieldIndex])
            clase = np.argmax(prob)
            resultado.push(clase)
                
            
