# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:09:39 2019

@author: Ahmert
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

from sklearn.metrics import confusion_matrix


def make_classifiers():
    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
    #log_reg = LogisticRegression()

    '''
    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]
    '''
    '''
    from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier
    classifiers = [ELMClassifier(n_hidden=30, rbf_width=0.01, random_state=0, alpha=0.1)]
    '''
    classifiers = [GenELMClassifier(hidden_layer=
                                    RBFRandomLayer(n_hidden=HIDDEN_NODE_COUNT, rbf_width=0.05, random_state=0))]
                                    
    return names, classifiers

'''
    Step Info per Node:
        [0]  : 0=will not fail, 1=will fail
        [1]  : init_load / avg_init_load
        [2]  : current_load / init_load
        [3]  : failed node count
        [4]  : avg distance of failed nodes
        [5]  : minimum distance of failed nodes
        [6]  : 1st degree neighbors count - shortest distance 1
        [7]  : 2nd degree neighbors count without 1st degree neighbors
        [8]  : average load of 1st degree neighbors
        [9]  : maximum load of 1st degree neighbors
        [10] : average load of 2nd degree neighbors
        [11] : maximum load of 2nd degree neighbors
        [12] : current network density
        -- GENERAL GRAPH INFO --
        [13] : initial graph density
        [14] : node count
        [15] : in sub graph connectivity probability
        [16] : inter sub graph connectivity probability
        [17] : gaussian mean connection
        [18] : gaussian standart deviation
        [19] : node tolerance percentage
        [20] : how many steps to fail, 0 if will not fail
'''
# Importing the dataset
matt = pd.read_csv('datasetBinary.csv', header=None)

# resample
#matt = matt.sample(frac=0.1, random_state=1)

# to replace greater than 4 distances with 5
#matt.loc[matt[9] > 5, 9] = 5
#matt.loc[matt[5] > 0, 5] = 0
#matt.loc[matt[4] > 0, 4] = 0
#matt.loc[matt[3] > 0, 3] = 0

'''
X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, 1:-7].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
'''
X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, 0:-1].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)

print('data generated')

#print(matt[20].value_counts())

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

HIDDEN_NODE_COUNT= 150
# ******************************** sci-kit ELM ********************************
names, classifiers = make_classifiers()

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    #score = clf.score(X_test, y_test)
    #print('Model %s score: %s' % (name, score))
    
    # Predicting the Test set results
    y_pred = clf.predict(X_test)
    org_pred = clf.predict(X_train)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    rsq_error = r2_score(y_test, y_pred) 
    rsq_error_org = r2_score(y_train, org_pred) 
    print('r^2 error: ' + str(rsq_error) + '(test) ' + str(rsq_error_org) + '(train)')
    
	# binary analysis
    #accu = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    #print('accuracy: ', accu)
    
    # label analysis
    totto = 0
    totok = 0
    totne = 0
    for x in range(6):
        for y in range(6):
            if x == y:
                totok += cm[x][y]
                totne += cm[x][y]
            if x-y == 1 or y-x == 1:
                totne += cm[x][y]
            totto += cm[x][y]
    print('total correct labels: ', str(totok/totto))
    print('all correct and near correct labels: ', str(totne/totto))


# ******************************** sci-kit ELM ********************************


