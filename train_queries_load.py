
import pdb

import csv, os, itertools, pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import mlflow, mlflow.tensorflow, logging

# mlflow.tensorflow.autolog(every_n_iter=2)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# user parameters
dataset = 'dataset 2'
maxNodes = 5
num_train_queries = 9000
num_test_queries = 1000
data_augm = False

input_path = dataset + '/training queries/'
filesList = os.listdir(input_path)

with open('X_train.pkl', 'rb') as handle:
    X_train = pickle.load(handle)
with open('X_test.pkl', 'rb') as handle:
    X_test = pickle.load(handle)
with open('y_train.pkl', 'rb') as handle:
    y_train = pickle.load(handle)
with open('y_test.pkl', 'rb') as handle:
    y_test = pickle.load(handle)
# X_train = pickle.load('X_train.pkl')
# X_test = pickle.load('X_test.pkl')
# y_train = pickle.load('y_train.pkl')
# y_test = pickle.load('y_test.pkl')

###################
with mlflow.start_run():
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # train on model

    history = model.fit( X_train, y_train, epochs = 10)

    # preds = model.predict(X_train)
    preds = model.predict(X_test)

    tot_err = 0
    for i in range(len(preds)):
        tot_err += ( preds[i] - y_test[i] ) **2
        
    MSE = tot_err[0] / len(preds)
    print(MSE)

    model.save(os.getcwd())
    
    # mlflow.log_param("loss", loss)
    # mlflow.tensorflow.log_model(model, "model")
    train_loss=history.history['loss'][-1]
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("MSE", MSE)
    
    mlflow.end_run()

    pdb.set_trace()