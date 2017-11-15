import numpy as np


class Datos(object):
    TiposDeAtributos = ('Continuo', 'Nominal')
    tipoAtributos = []
    nombreAtributos = []
    nominalAtributos = []
    datos = np.array(())

    SEP = ','
    # Lista de diccionarios. Uno por cada atributo.
    diccionarios = []

    # procesar el fichero para asignar correctamente las variables
    # tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        i = 1
        contador = 0.5
        n_recs = 0

        with open(nombreFichero, 'r') as file:
            for line in file:
                line = line.strip(' \t\n\r')
                if i == 1:
                    n_recs = int(line)
                elif i == 2:
                    self.nombreAtributos = line.split(self.SEP)
                    self.diccionarios = [{} for _ in range(len(self.nombreAtributos))]
                    self.datos = np.empty([n_recs, len(self.nombreAtributos)])
                elif i == 3:
                    types = np.asarray(line.split(self.SEP))
                    index_types = (types != 'Nominal') & (types != 'Continuo')
                    if sum(index_types) != 0:
                        raise ValueError('Tipo ' + types[index_types] + ' no soportado')

                    self.nominalAtributos = (types == 'Nominal')
                    self.tipoAtributos = types

                else:
                    data = line.split(self.SEP)
                    for index in range(len(data)):
                        if not self.nominalAtributos[index]:
                            self.datos[i - 4][index] = data[index]
                        else:
                            if data[index] in self.diccionarios[index]:
                                self.datos[i - 4][index] = self.diccionarios[index][data[index]]
                            else:
                                self.datos[i - 4][index] = contador
                                self.diccionarios[index][data[index]] = contador
                                contador += 1
                i += 1

            # Check for problems!:
            for indice_diccionario, diccionario in enumerate(self.diccionarios):
                for indice_key, key in enumerate(sorted(diccionario.keys())):
                    indices_actualizar = (self.datos[:, indice_diccionario] == diccionario[key])
                    self.datos[indices_actualizar, indice_diccionario] = float(indice_key)
                    diccionario[key] = float(indice_key)

    def extraeDatosTrain(self, idx):
        return np.take(self.datos, idx, axis=0)

    def extraeDatosTest(self, idx):
        return np.take(self.datos, idx, axis=0)

    def extraeDatosRelevantes(self,idx):
        return np.take(self.datos, idx, axis=1)

    def diccionarioRelevante(self,idx):
        return np.take(self.diccionarios, idx, axis=0)

    def atribDiscretosRelevante(self,idx):
        return np.take(self.nominalAtributos, idx, axis=0)

    def tipoAtribDiscretosRelevante(self,idx):
        return np.take(self.tipoAtributos, idx, axis=0)

