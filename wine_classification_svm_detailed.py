#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:45:26 2018

@author: amar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# =============================================================================
# Importing data
# =============================================================================

columns = ["Class", "Alcohol", "Malic acid", 
           "Ash", "Alcalinity of ash", "Magnesium", 
           "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
           "Proanthocyanins", "Color intensity", "Hue", 
           "OD280/OD315 of diluted wines", "Proline"]
wine = pd.read_csv('wine.data')
wine.columns = columns

# =============================================================================
# Separating data
# =============================================================================

wine_value = wine.values

rgen = np.random.RandomState(1)
rgen.shuffle(wine_value)

wine = pd.DataFrame(wine_value, columns = columns)

X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

train_size = 0.8

X_train = X[:int(len(X) * train_size), :]
X_test = X[int(len(X) * train_size):, :]
y_train = y[:int(len(y) * train_size)]
y_test = y[int(len(y) * train_size):]

plt.hist(X_train)
plt.show()

# =============================================================================
# Data Standardization
# =============================================================================
def mean(X):
    sum = 0
    for i in X:
        sum += i
    return sum / len(X)

def standard_deviation(X):
    sum = 0
    for i in X:
        sum += (i - mean(X)) ** 2
    return (sum / (len(X) - 1)) ** 0.5

        
X_train_std = (X_train - mean(X_train)) / standard_deviation(X_train)

plt.hist(X_train_std)
plt.show()
#plt.savefig("X_train_std_detailed.png")


# =============================================================================
# SVM classification with Grid-Search
# =============================================================================

param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.01, 0.1, 1, 10, 100, 1000]
        }


