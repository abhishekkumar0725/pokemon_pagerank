import pandas as pd
import numpy as np
from scipy import sparse

NUM_TYPES = 18

def readCSV():
    df = pd.read_csv('effectiveness.csv')
    return df

def buildAttackGraph(df):
    edges = []
    for index, row in df.iterrows():
        if row['Effectiveness'] == 2 and row['Attacking'] != row['Defending']:
            edge = [row['Defending'], row['Attacking'], 1]
            edges.append(edge)
    return edges

def buildDefendGraph(df):
    edges = []
    for index, row in df.iterrows():
        if row['Effectiveness'] == .5 and row['Attacking'] != row['Defending']:
            edge = [row['Attacking'], row['Defending'], 1]
            edges.append(edge)
    return edges

def typesDict():
    types = ['Normal','Fire','Water','Electric','Grass','Ice',
            'Fighting','Poison','Ground','Flying','Psychic','Bug',
            'Rock','Ghost','Dragon','Dark','Steel','Fairy']
    
    numberMap = {}
    for i, type in enumerate(types):
        numberMap[type] = i
    return numberMap

def pagerank(graph, p=.85):

    adjacencyMatrix= [[0]*NUM_TYPES for _ in range(NUM_TYPES)]
    translation = typesDict()

    for edge in graph:
        if edge[0] == 'A':
            print(edge)
        transition = translation[edge[0]]
        start = translation[edge[1]]
        adjacencyMatrix[start][transition] = 1
    
    adjacencyMatrix = np.array(adjacencyMatrix)
    A = sparse.csr_matrix(adjacencyMatrix,dtype=np.float)

    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]
    sink = rsums==0

    ro, r = np.zeros(NUM_TYPES), np.array([1/NUM_TYPES]*NUM_TYPES)
    while np.sum(np.abs(r-ro)) > .001:
        ro = r.copy()
        for i in range(0,NUM_TYPES):
            Ai = np.array(A[:,i].todense())[:,0]
            Di = sink / float(NUM_TYPES)
            Ei = np.ones(NUM_TYPES) / float(NUM_TYPES)
            r[i] = ro.dot( Ai*p + Di*p + Ei*(1-p) )
    return r/float(sum(r))

if __name__ == '__main__':
    df = readCSV()
    edges = buildAttackGraph(df)
    ranks = pagerank(graph=edges)
    print(ranks)
