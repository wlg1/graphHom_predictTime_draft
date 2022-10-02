import pdb

import csv, os, itertools, pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# user parameters
dataset = 'dataset 2'
maxNodes = 5
num_train_queries = 9000
num_test_queries = 1000
data_augm = False

input_path = dataset + '/training queries/'
filesList = os.listdir(input_path)
num_augm = []

for qID, file in enumerate(filesList[0:num_train_queries + num_test_queries]):
    queryFile = open(input_path + file, "r")
    
    node_labels = [0]*maxNodes
    edge_labels = {} #(x,y) : label
    # G = nx.DiGraph()
    
    allLines = queryFile.readlines()
    for line in allLines:
        line = line.replace('\n','')
        lineAsList = line.split(' ')
        if lineAsList[0] == 'v':
            # total_nodes = int(lineAsList[1]) + 1
            node_labels[int(lineAsList[1])] = int(lineAsList[2])
            # G.add_nodes_from([
            # (nodeID, {"label": nodeLabel}) ])
        elif lineAsList[0] == 'e':
            head = int(lineAsList[1])
            tail = int(lineAsList[2])
            edgeLabel = int(lineAsList[3])
            edge = (head, tail)
            edge_labels[edge] = int(edgeLabel) + 1
            # G.add_edge(head,tail, label = edgeLabel)
        
    X_sample = np.zeros((maxNodes, maxNodes))
    for edge in edge_labels.keys():
        X_sample[edge[0]][edge[1]] = edge_labels[edge]
        
    X_sample = X_sample.flatten()
    
    X_sample = np.append(X_sample, node_labels)
    
    X_sample = X_sample.reshape((1, len(X_sample)))
    if qID == 0:
        X_train = X_sample
    elif qID > 1 and qID < num_train_queries + 1:
        X_train = np.concatenate([X_train, X_sample])
    elif qID == num_train_queries + 1:
        X_test = X_sample
    elif qID > num_train_queries + 1 and qID < num_train_queries + 1 + num_test_queries:
        X_test = np.concatenate([X_test, X_sample])
        
    if data_augm:
        if (qID == num_train_queries + 1) or(qID > num_train_queries + 1 and qID < num_train_queries + 1 + num_test_queries):
            continue
        all_perms = list(itertools.permutations(node_labels))
        all_perms = [list(p) for p in all_perms]  #b/c list(set) treats (1,2) == (2,1)
        # b/c of repeated labels, check if redundant
        all_perms = list(set(all_perms))
        num_augm.append(len(all_perms) - 1)
        for permID, perm in enumerate(all_perms[1:]): #first one is the same
            print(qID, permID)
            X_sample = np.zeros((maxNodes, maxNodes))
            for edge in edge_labels.keys():
                new_head = perm[edge[0]]
                new_tail = perm[edge[1]]
                X_sample[new_head][new_tail] = edge_labels[edge]
            
            
            
            X_sample = X_sample.flatten()
            X_sample = np.append(X_sample, list(perm))
            X_sample = X_sample.reshape((1, len(X_sample)))
            X_train = np.concatenate([X_train, X_sample])
    
# pdb.set_trace()
with open('X_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)
with open('X_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)
    
###################

all_y = []
# resultsDir = dataset + '/training query results- CE and DE/'
resultsDir = dataset + '/output/'
for qID in range(0, num_train_queries + num_test_queries):
    print('parsing: ', qID)
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
            # prefiltTime.append(float(row[2]))
            simfiltTime.append(float(row[3]))
            buildTime.append(float(row[6]))
            # joinTime.append(float(row[7]))
            # totalTimes.append(float(row[8]))
            # numSolns[lblIndex] = row[-3].replace(',','')
    
    # totTime = prefiltTime[0] + simfiltTime[0] + buildTime[0]
    totTime = simfiltTime[0] + buildTime[0]
    all_y.append(totTime)
    
    if data_augm:
        if (qID == num_train_queries + 1) or(qID > num_train_queries + 1 and qID < num_train_queries + 1 + num_test_queries):
            continue
        all_y.extend([totTime]*num_augm[qID])
    
y_train = np.array(all_y[:num_train_queries])
y_test = np.array(all_y[num_train_queries:])
    
# pdb.set_trace()
with open('y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)
with open('y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

###################

model = Sequential()
model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# train on model

model.fit( X_train, y_train, epochs = 15000)

# preds = model.predict(X_train)
preds = model.predict(X_test)

tot_err = 0
for i in range(len(preds)):
    tot_err += ( preds[i] - y_test[i] ) **2
    
MSE = tot_err[0] / len(preds)
print(MSE)

pdb.set_trace()










