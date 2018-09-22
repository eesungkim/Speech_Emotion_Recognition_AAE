import os
import numpy as np

def normalize_Zscore(X_train, X_test):
    X=np.concatenate((X_train, X_test), axis=0)
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test

def normalize_MinMax(X_train, X_test):
    X=np.concatenate((X_train, X_test), axis=0)
    X_min = np.min(X, axis = 0)
    X_max = np.max(X, axis = 0)
    X_train = (X_train - X_min) / (X_max-X_min)
    X_test = (X_test - X_min) / (X_max-X_min) 
    return X_train, X_test
    
def dense_to_one_hot(labels_dense, num_classes=4):
    return np.eye(np.max(labels_dense)+1)[labels_dense]

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)