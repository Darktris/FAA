import numpy as np
from estrategiasparticionado.validacion_simple import ValidacionSimple

class PreprocesamientoAG:
    clasificador = None
    datos = None

    probabilidad_cruce = None
    probabilidad_mutacion_un_bit = None
    proporcion_elitismo = None
    max_generaciones = None
    max_fitness = None

    def __init__(self, probabilidad_cruce=0.6, probabilidad_mutacion_un_bit=0.001, proporcion_elitismo=0.05,
                 tamano_poblacion=50, max_generaciones=50, max_fitness=0.95):

        self.probabilidad_cruce = probabilidad_cruce
        self.probabilidad_mutacion_un_bit = probabilidad_mutacion_un_bit
        self.proporcion_elitismo = proporcion_elitismo
        self.tamano_poblacion = tamano_poblacion
        self.max_generaciones=max_generaciones
        self.max_fitness=max_fitness

    def seleccionarAtributos(self,dataset, clasificador):

        generaciones = 0
        fitness = 0.

        # Poblacion aleatoria
        poblacion = np.random.choice([True,False],(self.tamano_poblacion,dataset.datos.shape[1]-1))

        # mientras fitness < max_fitnes o generaciones < max_generaciones
        while fitness<self.max_fitness or generaciones< self.max_generaciones:

            # Seleccion progenitores
            puntuaciones = np.zeros(self.tamano_poblacion)
            # Evaluamos cada individuo de la poblacion
            for indice_individuo in range(puntuaciones.shape[0]):

                print("Individuo;",poblacion[indice_individuo])
                datos_relevantes = dataset.extraeDatosRelevantes(poblacion[indice_individuo])
                diccionario_relevante = dataset.extraeDatosRelevantes(poblacion[indice_individuo])

                estrategia = ValidacionSimple()

                clasificador.validacion(estrategia, dataset)

                puntuaciones[indice_individuo] = clasificador.errores[0]

            # Cruce
            # Mutacion
            # Seleccionar supervivientes
            print(puntuaciones)
            generaciones+=1
