import pandas as pd
import numpy as np
import sys

class Kmeans(object):
    """K-means """

    def __init__(self, arg):
        super(Kmeans, self).__init__()
        np.set_printoptions(edgeitems=20, linewidth=120, formatter=dict(float=lambda x: "%.3g" % x))#para dar
        #formato al output en la consola
        self.k = arg[1] #número de clústers #arg[0] es el nombre del archivo (kmeans.py)
        self.i = arg[2] #número de iteraciones (tipo k en k-fold?)
        self.r = arg[3] #número de veces que se ejecutará la identificación de agrupamientos
        self.errorProm = [] #arreglo con el valor promedio de las distancias de cada iteracion
        self.errorPorIteracion = []  #arreglo con los errores de cata iteracion(self.i)
        self.todosCentroides = []
        self.todosIdx = []
        self.todosClasificacion = []
        # cargar CSV
        self.dataFrame = pd.read_csv('iris.csv', header = None) #header->none. no tenemos una
                                                                #fila con los nombres de las columnas.
        self.dataset = self.dataFrame.to_numpy() #lo convierte a un numpy array
        self.datasetSinClases = self.dataset[:, :-1]
        print("k= " + self.k)
        print("i= " + self.i)
        print("r= " + self.r)
        print("\n")

    def initCentroids(self, iteracion): #datasetSinClases, k
        ###retorna k centroides iniciales
        self.centroids = np.zeros((int(self.k), int(self.dataset.shape[1])))
        self.centroidsSinClases = self.centroids[:,:-1]
        #Reordenar aleatoriamente las instancias
        auxArr = self.dataset
        np.random.shuffle(auxArr)
        #tomar las primeras k instancias
        self.centroids = auxArr[0:int(self.k) ,:]
        self.centroidsSinClases = self.centroids[:,:-1]

    def findClosestCentroids(self): #datasetSinClases, centroidsSinClases
        #crear un vector con los indices (valor del clúster asignado a cada instancia, 0 a k-1)
        errorTotal =0;
        self.idx = np.arange(self.dataset.shape[0])

        for iter in range(self.dataset.shape[0]): #por cada instancia
            #encontrar los valores minimos de las distancias contro los centroides
            term1 = np.subtract(self.datasetSinClases[iter,:], self.centroidsSinClases) #distancia
            #print("\n\nterm1 distancias**")
            #print(term1)
            term2 = np.square(term1)
            #print("\n\nterm2 distancias al cuadrado**")
            #print(term2)
            term3 = term2.sum(axis=1)#suma elementwise por filas, suma las distancias por cada
                                    #atributo en una sola distancia por centroide (distacias acumuladas por centroides)
            #print("\n\nterm3 distancias contra cada centroide**")
            #print(term3)
            term4 = np.amin(term3, axis=0)
            #print("\n\nterm4 distancia minima, instancia -> centroide")
            #print(term4)
            errorTotal += int(term4) #error acumulado de todas las instancias
            term5 = np.where(term3 == term4)
            term5 = term5[0][0]
            #print("\n\n term5 indice del centroide al que se tiene la menor distancia")
            #print(str(term5))
            self.idx[iter] = term5
        self.errorTotal = errorTotal

    def computeCentroids(self):
        for iter in range(int(self.k)):
            indicesCentroide = (self.idx == iter).nonzero()
            sumaX = self.datasetSinClases[indicesCentroide,:].sum(axis=1)
            cuentaIndices = len(indicesCentroide[0])
            self.centroidsSinClases[iter,:] = (1/cuentaIndices)*sumaX #movemos el centroide al promedio
        self.todosCentroides.append(self.centroidsSinClases)

    def run(self):
        menorError = 9000
        mejorIteracion = 99

        for i in range(int(self.i)):
            print("\niteración: " +str(i))
            self.initCentroids(i) #se inicializan los centroides con instancias aleatorias
            for iter in range(int(self.r)):
                self.findClosestCentroids() #calcular distancias y asignar centroides a cada instancia (idx)
                self.computeCentroids() #calcular y mover los centroides

            self.errorPorIteracion.append(self.errorTotal)
            print("error de la iteracion " + str(i) + ": "+ str(self.errorPorIteracion[i]))
            self.todosIdx.append(self.idx)
            print("idx iteracion " + str(i))
            print(str(self.todosIdx[i])) #esto es lo que hay que append a dataset sin clases
            print("\nCentroides finales de la iteracion " + str(i))
            print(str(self.centroidsSinClases))
            # self.datasetConClusters = np.column_stack((self.datasetSinClases, self.idx))
            self.datasetConClusters = np.column_stack((self.dataset, self.idx))
            if int(self.errorTotal < menorError):
                mejorIteracion = int(i)
                self.mejorResultado = self.datasetConClusters
                self.mejoresCentroides = self.centroidsSinClases #los centroides ya no tienen clases,
                                                #podemos calcularlas como la moda del cluster
                menorError = int(self.errorTotal)
        print("\n***********************************\n\t Mejores centroides:")
        print(self.mejoresCentroides)
        print("\n\t Error mas bajo:" + str(menorError))
        print("\n\tResultados de mejor iteración: (se muestran primeros 10)")
        print("\tatr1 \t\t| atr2 \t| atr3 \t| atr4 \t| #cluster asignado (0 a k)")
        print(self.mejorResultado[0:10,:])
        print("\n\nmejor iteracion: " + str(mejorIteracion))
        print("\n\n\n********** tabla completa de la mejor iteración **********")
        print("\n\n")
        print("\tatr1 \t\t| atr2 \t| atr3 \t| atr4 \t| #cluster asignado (0 a k)")
        print(self.mejorResultado)


if __name__ == '__main__':
    args = sys.argv
    kmeansObj = Kmeans(args)
    kmeansObj.run()
