import csv, os
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx

# input_path = 'training queries/'

# filesList = os.listdir(input_path)
# maxNodes = 5

# for qID, file in enumerate(filesList):
#     queryFile = open(input_path + file, "r")
    
#     node_labels = [0]*maxNodes
#     edge_labels = {} #(x,y) : label
#     # G = nx.DiGraph()
    
#     allLines = queryFile.readlines()
#     for line in allLines:
#         line = line.replace('\n','')
#         lineAsList = line.split(' ')
#         if lineAsList[0] == 'v':
#             # total_nodes = int(lineAsList[1]) + 1
#             node_labels[int(lineAsList[1])] = int(lineAsList[2])
#             # G.add_nodes_from([
#             # (nodeID, {"label": nodeLabel}) ])
#         elif lineAsList[0] == 'e':
#             head = int(lineAsList[1])
#             tail = int(lineAsList[2])
#             edgeLabel = int(lineAsList[3])
#             edge = (head, tail)
#             edge_labels[edge] = int(edgeLabel) + 1
#             # G.add_edge(head,tail, label = edgeLabel)
        
#     X_sample = np.zeros((maxNodes, maxNodes))
#     for edge in edge_labels.keys():
#         X_sample[edge[0]][edge[1]] = edge_labels[edge]
        
#     X_sample = X_sample.flatten()
    
#     X_sample = np.append(X_sample, node_labels)
    
#     X_sample = X_sample.reshape((1, len(X_sample)))
#     if qID == 0:
#         X_train = X_sample
#     else:
#         X_train = np.concatenate([X_train, X_sample])
    
###################

y_train = []

resultsDir = 'training query results- CE and DE/'
numQueries = 899
for qID in range(0, numQueries):
    print(qID)
    # impt to have '_' in _qi_ b/c q1 and q11
    filename = [f for f in listdir(resultsDir) if isfile(join(resultsDir, f))
                and '_q'+str(qID)+'_' in f]
    file = open(resultsDir + '/' + filename[0], 'r')
    
    reader = csv.reader(file, delimiter=',')
    startReadFlag = False
    totalTimes = []
    prefiltTime = []
    simfiltTime = []
    buildTime = []
    joinTime = []
    for row in reader:
        if not row:
            continue
        # if row[0] == 'id':
        if row[0] == 'Average':
            startReadFlag = True
            continue
        if not startReadFlag:
            continue
        if row[0].isnumeric() and len(row) > 11:  #to prevent it reading view times
            prefiltTime.append(float(row[2]))
            simfiltTime.append(float(row[3]))
            buildTime.append(float(row[6]))
            joinTime.append(float(row[7]))
            totalTimes.append(float(row[8]))
            # numSolns[lblIndex] = row[-3].replace(',','')
    
    totTime = prefiltTime[0] + simfiltTime[0] + buildTime[0]
    y_train.append(totTime)
    











