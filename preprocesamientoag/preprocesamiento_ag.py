import numpy as np
from estrategiasparticionado.validacion_simple import ValidacionSimple
from copy import deepcopy

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
        self.dataset_aux = None

    def seleccionarAtributos(self,dataset, clasificador):

        generaciones = 0
        fitness = 0.
        estrategia = ValidacionSimple()
        self.dataset_aux = deepcopy(dataset)


        # Poblacion aleatoria
        poblacion = np.random.choice([True,False],(self.tamano_poblacion,dataset.datos.shape[1]-1))

        # mientras fitness < max_fitnes o generaciones < max_generaciones
        while fitness<self.max_fitness or generaciones< self.max_generaciones:

            # Seleccion de progenitores (se reduce el tamano al 60% )
            indices_progenitores, puntuaciones = self.seleccion_progenitores(dataset,clasificador,poblacion,estrategia)

            tamano_elite = int(np.floor(self.tamano_poblacion*self.proporcion_elitismo))
            indices_elite = np.argpartition(puntuaciones,tamano_elite)[:tamano_elite]

            # Cruce
            sucesores = self.cruzar_progenitores(poblacion[indices_progenitores])

            # Mutacion
            mutantes = self.mutar(sucesores)

            # Seleccionar supervivientes
            # TODO
            generaciones+=1

    def seleccion_progenitores(self,dataset,clasificador,poblacion,estrategia):

        puntuaciones = np.zeros(self.tamano_poblacion)

        # Evaluamos cada individuo de la poblacion
        for indice_individuo in range(puntuaciones.shape[0]):
            print("Individuo;", poblacion[indice_individuo])

            datos_relevantes = dataset.extraeDatosRelevantes(poblacion[indice_individuo])
            diccionario_relevante = dataset.extraeDatosRelevantes(poblacion[indice_individuo])
            tipoAtributos_relevante = dataset.tipoAtribDiscretosRelevante(poblacion[indice_individuo])
            nominalAtributos_relevante = dataset.atribDiscretosRelevante(poblacion[indice_individuo])

            # Rellenamos el dataset auxiliar
            self.dataset_aux.datos = datos_relevantes
            self.dataset_aux.diccionario = diccionario_relevante
            self.dataset_aux.tipoAtributos = tipoAtributos_relevante
            self.dataset_aux.nominalAtributos = nominalAtributos_relevante

            clasificador.validacion(estrategia, self.dataset_aux)

            puntuaciones[indice_individuo] = clasificador.errores[0]

        weighted_randomizer = WeightedRandomizer(puntuaciones)
        tamano_progenitores = int(np.floor(self.probabilidad_cruce * self.tamano_poblacion))
        if tamano_progenitores % 2 != 0:
            tamano_progenitores -= 1

        return weighted_randomizer.random_array(tamano_progenitores), puntuaciones

    def cruzar_progenitores(self,progenitores):
        sucesores = np.ndarray(shape=progenitores.shape)

        indices_permutacion = np.random.permutation(progenitores.shape[0])

        for index in range(progenitores.shape[0] / 2):
            progenitor_1 = progenitores[indices_permutacion[2*index]]
            progenitor_2 = progenitores[indices_permutacion[2*index+1]]

            sucesores[2*index] = self.__cruzar_progenitores(progenitor_1,progenitor_2)
            sucesores[2*index+1] = self.__cruzar_progenitores(progenitor_1,progenitor_2)

        return sucesores


    def __cruzar_progenitores(self,progenitor_1,progenitor_2):

        v_a = np.random.choice([True,False],size=progenitor_1.shape[0])

        return (v_a and progenitor_1) or (not v_a and progenitor_2)

    def mutar(self,sucesores):
        mutantes = np.ndarray(shape=sucesores.shape)

        for index in range(sucesores.shape[0]):
            mutantes[index] = self.mutar_uno(sucesores[index])

        return mutantes

    def mutar_uno(self,sucesor):
        return np.fromiter(map(self.f_mut,sucesor),dtype=bool)

    def f_mut(self,bit):
        if np.random.uniform() <= self.probabilidad_mutacion_un_bit:
            return not bit
        else:
            return bit

class WeightedRandomizer:
    def __init__ (self, weights):
        self.__max = .0
        self.__weights = []

        for index, weight in enumerate(weights):
            self.__max += weight
            self.__weights.append ( (self.__max, index) )

    def random(self):
        r = np.random.random() * self.__max
        for weight, index in self.__weights:
            if weight > r:
                return index

    def random_array(self,size):
        return np.asarray([self.random() for _ in range(size)])
