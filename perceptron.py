import random
import math
import numpy as np

from dataset_iris import data as dt

dataSet = dt

#Clase para crear una neurona neuronas 
class Neurona:
    def __init__(self, nroEntradas, alpha=0.01):
        self.pesos = self.generarPesosIniciales(nroEntradas)
    
    def generarPesosIniciales(self, nroEntradas):
        pesos = []
        for entrada in range(nroEntradas+1):
            peso = random.uniform(-1.0, 1.0)
            pesos.append(peso)
        return pesos
    
    def feedForward(self, entradas):
        #Se inserta el Bias el la primera posición de las entradas
        entradas.insert(0, 1.0)
        #Promediar entradas ponderadas(Producto punto)
        ponderado = 0
        for i in range(len(entradas)):
            ponderado += self.pesos[i] * entradas[i]
        #Calcular el logaritmo ponderado(Solo aqui se puede usar numpy)
        exponencial= math.exp(-ponderado)
        valor = 1/ (1+ exponencial)
        return np.log(valor)
        
    def verPesos(self):
        print("[", end=" ")
        for i in range(len(self.pesos)):
            print(self.pesos[i], end=", ")
        print("]")

neurona  = Neurona(150)
neurona.verPesos()

#x=[]

"""for i in dataSet:
    if i == float:
        x = neurona.feedForward(dataSet)
print(x)"""

