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
        self.max_generaciones = max_generaciones
        self.max_fitness = max_fitness
        self.dataset_aux = None

        self.puntuacion = 0.
        self.atributos_seleccionados = None

    def seleccionarAtributos(self, dataset, clasificador,quiet=True):

        generaciones = 0
        fitness = 0.
        estrategia = ValidacionSimple()
        self.dataset_aux = deepcopy(dataset)

        # Poblacion aleatoria
        poblacion = np.random.choice([True, False], (self.tamano_poblacion, dataset.datos.shape[1] - 1))

        ### historic of indivuduals
        historic = []
        variabilidades = []
        historico_best_fits = []

        import time

        # mientras fitness < max_fitnes o generaciones < max_generaciones
        while fitness < self.max_fitness and generaciones < self.max_generaciones:

            # tiempo_base = time.clock()
            # Seleccion de progenitores (se reduce el tamano al 60% )
            indices_progenitores, puntuaciones = self.seleccion_progenitores(dataset, clasificador, poblacion,
                                                                             estrategia)

            # tiempo_evolucion = time.clock()

            indices_elite = np.argsort(-puntuaciones)
            elite = poblacion[indices_elite]

            self.puntuacion = puntuaciones[indices_elite][0]
            self.atributos_seleccionados = poblacion[indices_elite][0]

            # tiempo_seleccion = time.clock()

            # Cruce
            sucesores = self.cruzar_progenitores(poblacion[indices_progenitores])

            # tiempo_cruce = time.clock()

            # Mutacion
            mutantes = self.mutar(sucesores)

            # tiempo_mutacion = time.clock()

            # Construccion de la siguiente generacion
            nueva_generacion = np.ndarray(shape=poblacion.shape)

            numero_nuevos_individuos = mutantes.shape[0]
            numero_individuos_heredados = self.tamano_poblacion - numero_nuevos_individuos

            nueva_generacion[:numero_nuevos_individuos] = mutantes
            nueva_generacion[numero_nuevos_individuos:self.tamano_poblacion] = elite[:numero_individuos_heredados]

            # tiempo_construccion = time.clock()

            ######################################################################
            ######################################################################
            ######################################################################
            ######################################################################

            # print("######################################################################")
            # print("\nGeneracion:", generaciones+1)
            #
            # print("Tiempo evolucion:",tiempo_evolucion-tiempo_base)
            # print("Tiempo seleccion:",tiempo_seleccion-tiempo_evolucion)
            # print("Tiempo cruce:",tiempo_cruce-tiempo_seleccion)
            # print("Tiempo mutacion:",tiempo_mutacion-tiempo_cruce)
            # print("Tiempo construccion:",tiempo_construccion-tiempo_mutacion)
            #
            # for ranking, indice_individuo in enumerate(indices_elite):
            #     if list(poblacion[indice_individuo]) in historic:
            #         print("(Repetido) Rank:", 1+ranking,
            #               "Fitness:", puntuaciones[indice_individuo],
            #               "Hash", poblacion[indice_individuo].sum(),
            #               "Genotipo:", poblacion[indice_individuo][:5])
            #     else:
            #         historic.append(list(poblacion[indice_individuo]))
            #         print("Rank:", 1+ranking,
            #               "Fitness:", puntuaciones[indice_individuo],
            #               "Hash", poblacion[indice_individuo].sum(),
            #               "Genotipo:", poblacion[indice_individuo][:5])
            #
            # variabilidades.append(len(historic))
            # historico_best_fits.append(puntuaciones[indices_elite][0])
            # print("Variabilidad=", len(historic))

            ######################################################################
            ######################################################################
            ######################################################################
            ######################################################################

            if not quiet:
                print("Generacion:",generaciones+1,"Mediana puntuacion:",np.median(puntuaciones))

            # Actualizamos la poblacion con la nueva generacion
            poblacion = nueva_generacion

            generaciones += 1

        return poblacion, variabilidades, historico_best_fits

    def seleccion_progenitores(self, dataset, clasificador, poblacion, estrategia):

        puntuaciones = np.zeros(self.tamano_poblacion)

        # Evaluamos cada individuo de la poblacion
        for indice_individuo in range(puntuaciones.shape[0]):

            datos_relevantes = dataset.extraeDatosRelevantes(poblacion[indice_individuo])
            diccionarios_relevantes = dataset.diccionarioRelevante(poblacion[indice_individuo])
            tipoAtributos_relevantes = dataset.tipoAtribDiscretosRelevante(poblacion[indice_individuo])
            nominalAtributos_relevantes = dataset.atribDiscretosRelevante(poblacion[indice_individuo])

            # Rellenamos el dataset auxiliar
            self.dataset_aux.datos = datos_relevantes
            self.dataset_aux.diccionarios = diccionarios_relevantes
            self.dataset_aux.tipoAtributos = tipoAtributos_relevantes
            self.dataset_aux.nominalAtributos = nominalAtributos_relevantes

            # print("Individuo:",poblacion[indice_individuo])
            # print("Datos:",self.dataset_aux.datos.shape)
            # print("Dicc:",self.dataset_aux.diccionarios)
            # print("Tipos:",self.dataset_aux.tipoAtributos)
            # print("nominal:",self.dataset_aux.nominalAtributos)

            clasificador.validacion(estrategia, self.dataset_aux)

            puntuaciones[indice_individuo] = 1-clasificador.errores[0]

        weighted_randomizer = WeightedRandomizer(puntuaciones)

        tamano_progenitores = int(np.floor(self.probabilidad_cruce * self.tamano_poblacion))
        if tamano_progenitores % 2 != 0:
            tamano_progenitores -= 1

        return weighted_randomizer.random_array(tamano_progenitores), puntuaciones

    def cruzar_progenitores(self, progenitores):
        sucesores = np.ndarray(shape=progenitores.shape)

        indices_permutacion = np.random.permutation(progenitores.shape[0])
        max_indice = int(np.floor(progenitores.shape[0] / 2))

        for index in range(max_indice):
            progenitor_1 = progenitores[indices_permutacion[2 * index]]
            progenitor_2 = progenitores[indices_permutacion[2 * index + 1]]

            sucesores[2 * index] = cruzar_dos_progenitores(progenitor_1, progenitor_2)
            sucesores[2 * index + 1] = cruzar_dos_progenitores(progenitor_2, progenitor_1)

        return sucesores

    def mutar(self, sucesores):
        mutantes = np.ndarray(shape=sucesores.shape)

        for index in range(sucesores.shape[0]):
            mutantes[index] = self.mutar_uno(sucesores[index])

        return mutantes

    def mutar_uno(self, sucesor):
        return np.fromiter(map(self.f_mut, sucesor), dtype=bool)

    def f_mut(self, bit):
        if np.random.uniform() <= self.probabilidad_mutacion_un_bit:
            return not bit
        else:
            return bit

def cruzar_dos_progenitores(progenitor_1, progenitor_2):

    v_a = np.random.choice([True,False], size=progenitor_1.shape[0])

    return np.logical_or(np.logical_and(v_a,progenitor_1),np.logical_and(np.logical_not(v_a),progenitor_2))

class WeightedRandomizer:
    def __init__(self, weights):
        self.__max = .0
        self.__weights = []

        for index, weight in enumerate(weights):
            self.__max += weight
            self.__weights.append((self.__max, index))

    def random(self):
        r = np.random.random() * self.__max
        for weight, index in self.__weights:
            if weight > r:
                return index

    def random_array(self, size):
        return np.asarray([self.random() for _ in range(size)])
