# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 10:32:58 2019

@author: Ahmert
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

'''
	data transform 
		[0]  : init_efficiency / avg_init_efficiency						(0+)
		[1]  : current_efficiency / init_efficiency							(0..1)
		[2]  : failed node count											(0+)
		[3]  : 1st degree neighbors count - shortest distance 1				(0+)
		[4]  : 2nd degree neighbors count without 1st degree neighbors		(0+)
		[5]  : minimum efficiency of 1st degree neighbors					(0..1)
		[6]  : average efficiency of 1st degree neighbors					(0..1)
		[7]  : current network density										(0..1)
		[8]  : initial graph density										(0..1)
		[9]  : 0=will not fail, 1=will fail		||		step distance to fail
		
'''
# Importing the dataset
matt = pd.read_csv('./GR_TEST/indiv_summ.stxt', header=None, delim_whitespace=True)
#matt = pd.read_csv('datasetBinary.csv', header=None)
#matt = matt.sample(frac=0.1, random_state=1)

# to replace greater than 4 distances with 5
#matt.loc[matt[20] > 10, 20] = 10
#matt.loc[matt[5] > 0, 5] = 0
#matt.loc[matt[4] > 0, 4] = 0
#matt.loc[matt[3] > 0, 3] = 0

X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, [1,2,3,6,7,8,9,12,13]].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
#    matt.iloc[:, 0:-1].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
#	matt.iloc[:, [0,1,3,4,5,6,8]].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
#	matt.iloc[:, [0,1,3,4,5,6,7,8]].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)

print('data generated')

#print(matt[20].value_counts())

# ********************************** options **********************************
SINGLE_OP = True
to_save = False
from_save = False
# ********************************** options **********************************

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
'''
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
'''
if SINGLE_OP == False:
    dict_test = {}
    dict_train = {}
    for num in range(35):
        if from_save:
            #classifier = pickle.load(open('theModel.mdl', 'rb'))
            classifier = joblib.load('theModel.mdl')
        else:
            classifier = RandomForestClassifier(n_estimators = 3*num+1, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        org_pred = classifier.predict(X_train)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        rsq_error = r2_score(y_test, y_pred) 
        rsq_error_org = r2_score(y_train, org_pred) 
        print((3*num+1), ': r^2 error: ' + str(rsq_error) + '(test) ' + str(rsq_error_org) + '(train)')
        
        dict_test[3*num+1] = rsq_error
        dict_train[3*num+1] = rsq_error_org
    
    '''
    plt.bar(range(len(dict_)), list(dict_.values()), align='center')
    plt.xticks(range(len(dict_)), list(dict_.keys()))
    '''
    plt.plot(range(len(dict_test)), np.array(list(dict_test.values())))
    plt.plot(range(len(dict_train)), np.array(list(dict_train.values())))
    
    plt.show()

else :
    if from_save:
        #classifier = pickle.load(open('theModel.mdl', 'rb'))
        classifier = joblib.load('theModel.mdl')
    else:
        #classifier = RandomForestClassifier(n_estimators = 30, criterion='entropy', random_state=0)
        classifier = RandomForestClassifier(n_estimators = 20, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    org_pred = classifier.predict(X_train)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    rsq_error = r2_score(y_test, y_pred) 
    rsq_error_org = r2_score(y_train, org_pred) 
    print('r^2 error: ' + str(rsq_error) + '(test) ' + str(rsq_error_org) + '(train)')
    
    if to_save:
        joblib.dump(classifier, 'theModel.mdl')
        

# binary analysis
#accu = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
#print('accuracy: ', accu)

# label analysis
totto = 0
totok = 0
totne = 0
for x in range(11):
    for y in range(11):
        if x == y:
            totok += cm[x][y]
            totne += cm[x][y]
        if x-y == 1 or y-x == 1:
            totne += cm[x][y]
        totto += cm[x][y]
print('total correct labels: ', str(totok/totto))
print('all correct and near correct labels: ', str(totne/totto))

