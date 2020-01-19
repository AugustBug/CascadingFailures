# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:02:44 2019

@author: Ahmert
"""


from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import cv2
import sys
import os
import pandas as pd

#from libLstm import *


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score



nInputTextOneHot = 11
nLstmFeatures=nInputTextOneHot

n_steps = 1
n_outputs = 1
nOutputs = n_outputs
dataSetBatchSize = 512
n_features = 9
n_input = n_features
nOutputSeq = 1
n_hidden = 256



# Importing the dataset
matt = pd.read_csv('datasetDist.csv', header=None, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(\
    matt.iloc[:, 0:-1].values, matt.iloc[:, -1].values, test_size=0.1, random_state = 0)
	#matt.iloc[:, 0:-1].values, matt.iloc[:, -1].values, test_size=0.3, random_state = 0)
    #matt.iloc[0:1000000, 0:-1].values, matt.iloc[0:1000000, -1].values, test_size=0.3, random_state = 0)
	
print('data generated')



# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, nInputTextOneHot])

np.random.seed(1)
tf.set_random_seed(1)


# Define weights
weights = {
#    'out': tf.Variable(tf.random_normal([lb.n_hidden, n_outputs]))

        #'out': tf.Variable(tf.random_normal([lb.n_hidden, 16], mean=1.0, stddev=0.0000001))
        'out': tf.Variable(tf.random_normal([n_hidden, nLstmFeatures* nOutputSeq]))

}
biases = {
    #'out': tf.Variable(tf.random_normal([n_outputs],mean=0.0, stddev=0.00001,))

#   'out': tf.Variable(tf.random_normal([16]))
    'out': tf.Variable(tf.random_normal([nLstmFeatures* nOutputSeq],mean=0.0, stddev=0.00001))

        
}






'''

# Define the LSTM cells
lstm_cells = [rnn.LSTMCell(lb.n_hidden,activation=tf.nn.leaky_relu) for _ in range(lb.n_layers)]
stacked_lstm = rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=x, dtype=tf.float32)

h = tf.transpose(outputs, [1, 0, 2])
pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])
'''


xFlat = tf.reshape(x, [-1,n_features])
    
# Define Dense object which is reusable
dense1 = tf.layers.Dense(units=32, activation=tf.nn.leaky_relu)
dense2 = tf.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
dense3 = tf.layers.Dense(units=32, activation=tf.nn.relu)

logits = tf.layers.Dense(units=nInputTextOneHot, activation=tf.nn.leaky_relu)

d1 = dense1(xFlat)
d2 = dense2(d1)
d3 = dense3(d2)


output = logits(d3)



logit = tf.reshape(output, [-1,nInputTextOneHot])

#text cost
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y ))
#cost = tf.reduce_mean(tf.squared_difference(logit, y ))




logitArgmax = tf.argmax(logit,axis=1)
yArgmax = tf.argmax(y,axis=1)

cost0 = tf.Print(cost,[logitArgmax[:],'                 ', yArgmax[:],'             ' ,tf.shape(yArgmax)] , '\nprinting yArgmax: ', summarize=50)



correctPredCount = tf.reduce_sum(tf.cast(tf.equal(logitArgmax, yArgmax),tf.float32))

acc = tf.divide(correctPredCount,dataSetBatchSize)


'''
logitSig = tf.round(tf.nn.sigmoid(logit))



correctPredCount = tf.reduce_sum(tf.cast(tf.equal(logitSig, y),tf.float32))

acc = tf.divide(correctPredCount,dataSetBatchSize)
'''

costP = tf.Print(cost0,[y[:],'                 ', logit[:],'             ',  cost, '    ' ,tf.shape(y)] , '\nprinting cost: ', summarize=50)


#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1,beta1=0.8,beta2=0.2).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)







print('declaretion is done..')

# Initializing the variables
init = tf.global_variables_initializer()



print('initialization is done..')


gpu_options = tf.GPUOptions(allow_growth=True)


# Launch the graph
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    step = 0

    loss_value = 99999999999999
    target_loss = 0.15
    #target_loss = 0.24
    #target_loss = 10000000.15
    
    training_iters = 10000000000000
    
    epoch = 0
    nIter = 0
    
    
    nSample = X_train.shape[0]
    
    y_train = np.eye(nInputTextOneHot)[y_train.astype(np.int32)]
    y_test = np.eye(nInputTextOneHot)[y_test.astype(np.int32)]
    
   
            
    # Keep training until we reach max iterations
    while step < training_iters and loss_value > target_loss:
        
        
        if step*dataSetBatchSize > nSample:
            stepMod = 0
        else:
            stepMod = step
            
        
        start = stepMod*dataSetBatchSize
        end = start + dataSetBatchSize
        batch_x = np.reshape(X_train[start:end] , (-1,n_steps,n_features) )
        batch_y = np.reshape(y_train[start:end] , (-1,nInputTextOneHot) )
        
        sess.run([optimizer], feed_dict={x: batch_x, y: batch_y})
        
       
        
        #if step % (lb.display_step*100) == 0:
        if step % (1000*1) == 0:
            loss_value, accVal = sess.run([cost, acc], feed_dict={x: batch_x, y: batch_y})
            #loss_value = sess.run(totalCost, feed_dict={x: batch_x, y: batch_y})
            
       
            print("Iter " + str(step*dataSetBatchSize) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss_value)+ " acc: " + "{:.6f}".format(accVal))
            
        
            
            sess.run(costP, feed_dict={x: batch_x, y: batch_y})
            
         
                
        step += 1
        nIter = step * dataSetBatchSize
    

            
    print("Optimization Finished!")
    
    
    
    
    
    batchTest_x = np.reshape(X_test, (-1,n_steps, n_features))
    batchTest_y = np.reshape(y_test, (-1,nInputTextOneHot))
            
    print(batchTest_x.shape)
    print(batchTest_y.shape)
    
    
    
    accTest = sess.run([acc], feed_dict={x: batchTest_x, y: batchTest_y})
    
    print('Test acc: ' , accTest)
    
    
    #pred = tf.round(tf.nn.sigmoid(logit)).eval({x: np.array(batchTest_x), y: np.array(batchTest_y)})
    pred = tf.round( tf.argmax(logit,axis=1)).eval({x: batchTest_x, y: batchTest_y})
    
    '''
    f1 = f1_score(np.array(batchTest_y), pred, average='macro')
    accuracy=accuracy_score(np.array(batchTest_y), pred)
    recall = recall_score(y_true=np.array(batchTest_y), y_pred= pred)
    precision = precision_score(y_true=np.array(batchTest_y), y_pred=pred)
    

    print("F1 Score:", f1)
    print("Accuracy Score:",accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    '''

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    y_test = np.argmax(y_test, axis=1)
    #pred = np.argmax(pred, axis=1)
    
    cm = confusion_matrix(y_test, pred)

    print(cm)
   
    

sess.close()
print ('Session closed..')