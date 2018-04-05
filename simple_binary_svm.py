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
# Separating data into 2 classes
# =============================================================================

wine_value = wine.values
wine = pd.DataFrame(wine_value, columns = columns)

X, y = wine.iloc[:, 1:3].values, wine.iloc[:, 0].values
X = X[:len(y[y != 3])]
y = y[:len(y[y != 3])]

print("Number of unique classes:")
print(np.unique(y))

# =============================================================================
# Class mapping
# =============================================================================
y[y == 1] = -1
y[y == 2] = 1

#=============================================================================
# Shuffling data
# =============================================================================
X = np.column_stack((X, y))
rgen = np.random.RandomState(1)
rgen.shuffle(X)
y = X[:, 2]
X = X[:, :2]

# =============================================================================
# Seperating data into train and test sets
# =============================================================================

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

# =============================================================================
# Data plot
# =============================================================================
plt.scatter(X_train_std[y_train == -1, 0], X_train_std[y_train == -1, 1], label = "Class 0", marker = "x")
plt.scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], label = "Class 1", marker = "v")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()

# =============================================================================
# Weight initialization
# =============================================================================
random_state = 1
rgen = np.random.RandomState(random_state)
weight = rgen.normal(loc = 0.0, scale = 0.01, size = X_train_std.shape[1] + 1)

# =============================================================================
# Learning rate and epoch
# =============================================================================
eta = 0.001
epoch = 100

# =============================================================================
# SVM with gradient descent
# Cost function is SSE(Sum of Squared Errors)
# =============================================================================
for i in range(epoch):
    output = y_train * (weight[0] + np.dot(X_train_std, weight[1:]))
    output = output >= 1
    if np.unique(output).all() != True:
        net_input = np.dot(X_train_std, weight[1:]) + weight[0]
        errors = y_train - net_input
        weight[1:] += eta * X_train_std.T.dot(errors)
        weight[0] = eta * errors.sum()
    else:
        print(weight)
        break

# =============================================================================
# Plot the decision boundary
# =============================================================================
t = np.arange(-2, 2)
plt.plot(t, -(weight[0] + weight[1] * t)/weight[2])
#plt.savefig('graph1.png')
plt.show()
