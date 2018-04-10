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
wine = pd.read_csv('wine_train.data')
wine.columns = columns

# =============================================================================
# Separating data into 2 classes
# =============================================================================

wine_value = wine.values
wine = pd.DataFrame(wine_value, columns = columns)

X, y = wine.iloc[:, 1:3].values, wine.iloc[:, 0].values

X = np.array([[2.0, 6.0], [4.0, 8.0], [6.0, 1.0], [8.0, 1.0], [10.0, 7.0], [12.0, 5.0]])
y = np.array([1, 1, 2, 2, 3, 3])
#X = X[:len(y[y != 3])]
#y = y[:len(y[y != 3])]

number_of_classes = len(np.unique(y))
print("Number of unique classes:", number_of_classes)
random_state = 1
# =============================================================================
# Class mapping takes only two classes
# First class = -1
# Second class = 1
# =============================================================================
def class_mapping(y):
    y[y == np.unique(y)[0]] = -1
    y[y == np.unique(y)[1]] = 1
    return y
#=============================================================================
# Shuffling data
# =============================================================================
X = np.column_stack((X, y))
rgen = np.random.RandomState(random_state)
rgen.shuffle(X)
y = X[:, 2]
X = X[:, :2]

# =============================================================================
# Seperating data into train and test sets
# =============================================================================

train_size = 0.8

X_train = X[:round(len(X) * train_size), :]
X_test = X[round(len(X) * train_size):, :]
y_train = y[:round(len(y) * train_size)]
y_test = y[round(len(y) * train_size):]

#plt.hist(X_train)
#plt.show()

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

        
#X_train_std = (X_train - mean(X_train)) / standard_deviation(X_train)
#X_test_std = (X_test - mean(X_train)) / standard_deviation(X_train)
        
X_train_std = X_train
X_test_std = (X_test - mean(X_train)) / standard_deviation(X_train)


#plt.hist(X_train_std)
#plt.show()
#plt.savefig("X_train_std_detailed.png")


# =============================================================================
# SVM classification with Grid-Search
# =============================================================================

# =============================================================================
# Data plot
# =============================================================================
#plt.scatter(X_train_std[y_train == -1, 0], X_train_std[y_train == -1, 1], label = "Class 0", marker = "x")
#plt.scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], label = "Class 1", marker = "v")
#plt.xlabel("Alcohol")
#plt.ylabel("Malic Acid")
#plt.legend()


# =============================================================================
# Learning rate and epoch
# =============================================================================
eta = 0.1
epoch = 10

# =============================================================================
# Function of SVM with gradient descent
# Cost function is SSE(Sum of Squared Errors)
# =============================================================================
def svm(X, y, weight):
    for i in range(epoch):
        output = y * (weight[0] + np.dot(X, weight[1:]))
        output = output >= 1
        if np.unique(output).all() != True:
            net_input = np.dot(X, weight[1:]) + weight[0]
            errors = y - net_input
            weight[1:] += eta * X.T.dot(errors)
            weight[0] = eta * errors.sum()
        else:
            print(weight)
            break
        return weight
# =============================================================================
# Number of classifiers
# =============================================================================
number_of_classifiers = number_of_classes * (number_of_classes - 1) / 2.0
print("Number of classifiers: ", number_of_classifiers)

# =============================================================================
# Weight initialization
# =============================================================================
rgen = np.random.RandomState(random_state)
weight = rgen.normal(loc = 0.0, scale = 0.01, size = (X_train_std.shape[1] + 1, number_of_classes))

# =============================================================================
# Classification
# =============================================================================
weight_iteration = 0
for i in range(number_of_classes):
    for j in range(i + 1, number_of_classes):
        train_batch = np.column_stack((X_train_std, y_train))
        idx = np.array([i, j]) + 1
        train_batch = train_batch[train_batch[:, 2] != int(np.setdiff1d(np.unique(y), idx))]
        train_batch_X = train_batch[:, :2]
        train_batch_y = train_batch[:, 2]
        mapped_y = y_train[y_train != i + 1]
        mapped_y = class_mapping(mapped_y)
        
        weight[weight_iteration] = svm(train_batch_X, train_batch_y, weight[weight_iteration])
        weight_iteration += 1
print(weight)

def plot_decision_boundary(weight):
    t = np.arange(0, 10)
    plt.plot(t, -(weight[0] + weight[1] * t)/weight[2])
    #plt.savefig('graph1.png')
for i in range(len(weight)):
    plot_decision_boundary(weight[i])
    
plt.scatter(X_train_std[y_train == 1, 0], X_train_std[y_train == 1, 1], label = "Class 0", marker = "x")
plt.scatter(X_train_std[y_train == 2, 0], X_train_std[y_train == 2, 1], label = "Class 1", marker = "v")
plt.scatter(X_train_std[y_train == 3, 0], X_train_std[y_train == 3, 1], label = "Class 2", marker = "*")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()

    
plt.show()
    
# =============================================================================
# Plot the decision boundary
# =============================================================================

    
