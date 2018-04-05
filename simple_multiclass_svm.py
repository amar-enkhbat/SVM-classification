#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:13:04 2018

@author: amar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Data import
# =============================================================================

columns = ["Class", "Alcohol", "Malic acid", 
           "Ash", "Alcalinity of ash", "Magnesium", 
           "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
           "Proanthocyanins", "Color intensity", "Hue", 
           "OD280/OD315 of diluted wines", "Proline"]
wine = pd.read_csv('wine.data')
wine.columns = columns

X, y = wine.iloc[:107, 1:3].values, wine.iloc[:107, 0].values

# =============================================================================
# Data mapping
# =============================================================================
class_length = len(y[y == 1]) 

y[:class_length] = -1
y[class_length:] = 1

# =============================================================================
# Data standardization
# =============================================================================
X_std = np.copy(X.astype(float))
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# =============================================================================
# Data plot
# =============================================================================
plt.scatter(X_std[:48, 0], X_std[:48, 1], label = "Class 0", marker = "x")
plt.scatter(X_std[48:, 0], X_std[48:, 1], label = "Class 1", marker = "v")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()

# =============================================================================
# Weight initialization
# =============================================================================
random_state = 1
rgen = np.random.RandomState(random_state)
weight = rgen.normal(loc = 0.0, scale = 0.01, size = X_std.shape[1] + 1)

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
    output = y * (weight[0] + np.dot(X_std, weight[1:]))
    output = output >= 1
    if np.unique(output).all() != True:
        net_input = np.dot(X_std, weight[1:]) + weight[0]
        errors = y - net_input
        weight[1:] += eta * X_std.T.dot(errors)
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