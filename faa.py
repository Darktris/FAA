import numpy as np

class Datos(object):
  
    TiposDeAtributos=('Continuo','Nominal')
    tipoAtributos=[]
    nombreAtributos=[]
    nominalAtributos=[]
    datos=np.array(())
    SEP = ','
    # Lista de diccionarios. Uno por cada atributo.
    diccionarios=[]
 
    # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        i = 1
        contador = 0
        with open(nombreFichero, 'r') as file:
            for line in file:
                if i == 1:
                    n_recs = int(line.strip(' \t\n\r'))
                elif i == 2:
                    self.nombreAtributos = line.split(self.SEP)
                    self.diccionarios = [{} for _ in range(len(self.nombreAtributos)) ] 
                    self.datos = np.empty([n_recs, len(self.nombreAtributos)], dtype=int)
                elif i == 3:
                    types = line.split(self.SEP)
                    for _type in types:
                        _type = _type.strip(' \t\n\r')
                        if _type.strip() not in self.TiposDeAtributos:
                            raise ValueError('Tipo ' + _type + ' no soportado')
                    self.tipoAtributos = types
                else:
                    data = line.split(self.SEP)
                    for index in range(len(data)):
                        if self.tipoAtributos[index] == 'Continuo':
                            self.datos[i-4][index] = data[index].strip(' \t\n\r')
                        else:
                            if data[index].strip(' \t\n\r') in self.diccionarios[index]:
                                self.datos[i-4][index] = self.diccionarios[index][data[index].strip(' \t\n\r')]
                            else:
                                self.datos[i-4][index] = contador
                                self.diccionarios[index][data[index].strip(' \t\n\r')] = contador
                                contador += 1
                    #self.datos[i-4] = data 
                i += 1
            
            # Correcion de los valores del diccionario
            for i in range(len(self.diccionarios)):
                contador = 0
                for key in sorted(self.diccionarios[i].keys()):
                    
                    copia = np.array(self.datos[:,i],copy=True)
                    
                    for j in range(len(copia)):
                        if copia[j] == self.diccionarios[i][key]:
                            self.datos[j,i] = contador
                    self.diccionarios[i][key] = contador
                    contador += 1
    
    # TODO: implementar en la práctica 1
    def extraeDatosTrain(idx):
        pass
  
    def extraeDatosTest(idx):
        pass


  