import pandas as pd
import numpy as np
from scipy import sparse

NUM_TYPES = 18**2

def readCSV():
    df = pd.read_csv('effectiveness.csv')
    return df

def dualCSV():
    df = pd.read_csv('dualTyping.csv')
    return df

def dualTyping():
    df = readCSV()
    df2 = readCSV()
    dualDF = pd.DataFrame(columns=['Attacking Type 1', 'Attacking Type 2',
                                    'Defending Type 1', 'Defending Type 2',
                                    'Effectiveness'])

    for index, type1 in df.iterrows():
        for index, type2 in df2.iterrows():
            effectiveness = type1['Effectiveness'] * type2['Effectiveness']
            row = { 'Attacking Type 1': type1['Attacking'],#min(type1['Attacking'], type2['Attacking']),
                    'Attacking Type 2': type2['Attacking'],#max(type1['Attacking'], type2['Attacking']),
                    'Defending Type 1': type1['Defending'],#min(type1['Defending'], type2['Defending']),
                    'Defending Type 2': type2['Defending'],#max(type1['Defending'], type2['Defending']),
                    'Effectiveness': effectiveness}
            dualDF = dualDF.append(row, ignore_index=True)
    
    #dualDF = dualDF.drop_duplicates(keep='first').reset_index(drop=True)
    dualDF.to_csv('dualTyping.csv', encoding='utf-8', index=False)

def buildAttackGraph(df):
    edges = []
    for index, row in df.iterrows():
        attack = row['Attacking Type 1'] + row['Attacking Type 2']
        defend = row['Defending Type 1'] + row['Defending Type 2']
        if row['Effectiveness'] > 1 and attack != defend:
            edge = [attack, defend, row['Effectiveness']/2]
            edges.append(edge)
    return edges

def buildDefendGraph(df):
    edges = []
    for index, row in df.iterrows():
        attack = row['Attacking Type 1'] + row['Attacking Type 2']
        defend = row['Defending Type 1'] + row['Defending Type 2']
        if row['Effectiveness'] < 1 and attack != defend:
            edge = [defend, attack, row['Effectiveness']*2]
            edges.append(edge)
    return edges

def typesDict():
    types = ['Normal','Fire','Water','Electric','Grass','Ice',
            'Fighting','Poison','Ground','Flying','Psychic','Bug',
            'Rock','Ghost','Dragon','Dark','Steel','Fairy']
    
    dualTypes = []
    for type1 in types:
        for type2 in types:
            dualTypes.append(type1+type2)

    numberMap = {}
    for i, type in enumerate(dualTypes):
        numberMap[type] = i
    return numberMap

def pagerank(graph, p=.85):

    adjacencyMatrix= [[0]*NUM_TYPES for _ in range(NUM_TYPES)]
    translation = typesDict()

    for edge in graph:
        transition = translation[edge[0]]
        start = translation[edge[1]]
        adjacencyMatrix[start][transition] = edge[2]
    
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
    df = dualCSV()
    edges = buildAttackGraph(df)
    print(pagerank(edges))
