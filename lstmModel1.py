# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 03:30:15 2019

@author: Ahmert
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

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
def recurrent_neural_network_model():
    layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.split(xplaceholder, n_features, 1)
    print(x)
    lstm_cell = rnn.BasicLSTMCell(n_units)    
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)   
    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']
    return output

def train_neural_network():
    logit = recurrent_neural_network_model()
    logit = tf.reshape(logit, [-1])
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            numo = int(len(X_train) / batch_size)
            for i in range(numo):
                start = i * batch_size
                end = start + batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])                
                _, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
                epoch_loss += c
                #i += batch_size
            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
        pred = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(X_test), yplaceholder: np.array(y_test)})
        f1 = f1_score(np.array(y_test), pred, average='macro')
        accuracy=accuracy_score(np.array(y_test), pred)
        recall = recall_score(y_true=np.array(y_test), y_pred= pred)
        precision = precision_score(y_true=np.array(y_test), y_pred=pred)

        print("F1 Score:", f1)
        print("Accuracy Score:",accuracy)
        print("Recall:", recall)
        print("Precision:", precision)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred)

        print(cm)



# Importing the dataset
matt = pd.read_csv('datasetBinary.csv', header=None)

'''
matt = pd.read_csv('./GR_TEST/indiv_summ.stxt', header=None, delim_whitespace=True)

# to replace greater than 4 distances with 5
matt.loc[matt[20] > 10, 20] = 10
#matt.loc[matt[5] > 0, 5] = 0
#matt.loc[matt[4] > 0, 4] = 0
#matt.loc[matt[3] > 0, 3] = 0
'''
'''
X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, 1:-7].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
'''

matt = matt.sample(frac=0.1, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, 0:-1].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
print('data read')
'''
X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, [1,2,3,6,7,8,9,12,13]].values, matt.iloc[:, 0].values, test_size=0.3, random_state = 0)

print('data generated')

print(matt[20].value_counts())
'''	
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# *********************************** LSTM ***********************************

# hyperparameters
epochs = 100
n_classes = 1
n_units = 100
n_features = 9
batch_size = 50

xplaceholder= tf.placeholder('float',[None,n_features])
yplaceholder = tf.placeholder('float')

train_neural_network()

'''
classifier.fit(X_train, y_train)
#score = clf.score(X_test, y_test)
#print('Model %s score: %s' % (name, score))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
org_pred = classifier.predict(X_train)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

rsq_error = r2_score(y_test, y_pred) 
rsq_error_org = r2_score(y_train, org_pred) 
print('r^2 error: ' + str(rsq_error) + '(test) ' + str(rsq_error_org) + '(train)')


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
'''
# *********************************** LSTM ***********************************