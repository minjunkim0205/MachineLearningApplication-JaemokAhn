# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:53:47 2024

@author: ajm
"""

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def read(filename, date_idx, date_parse, year=None, bucket=7):
    days_in_year = 365
    
    freq = {}
    if year != None:
         for period in range(0, int(days_in_year / bucket)):
            freq[period] = 0
        
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[date_idx] == '':
                continue
            
            t = time.strptime(row[date_idx], date_parse)
            if year == None:
                if not t.tm_year in freq:
                    freq[t.tm_year] = {}
                    for period in range(0, int(days_in_year / bucket)):
                        freq[t.tm_year][period] = 0
                
                if t.tm_yday < (days_in_year - 1):
                    freq[t.tm_year][int(t.tm_yday / bucket)] += 1
                    
            else:
                if t.tm_year == year and t.tm_yday < (days_in_year-1):
                    freq[int(t.tm_yday / bucket)] += 1
    
    return freq

freq = read(r'C:\Users\dahae\machine learning\311_call_center.csv' , 1, '%m/%d/%Y %H:%M:%S %p', 2014)

X_train = np.asarray(list(freq.keys()))
Y_train = np.asarray(list(freq.values()))
#plt.scatter(X_train,Y_train)

maxY = np.max(Y_train)
nY_train = Y_train / np.max(Y_train)
#plt.scatter(X_train,nY_train)

class Model:
    def __init__(self):
        self.mu = tf.Variable(1.0, dtype=tf.float32)
        self.sig = tf.Variable(1.0, dtype=tf.float32)
    def __call__(self, x):
        x_c = tf.cast(x, tf.float32)
        return tf.exp(-tf.pow(x_c - self.mu, 2.) / (2. * tf.pow(self.sig, 2.)))

def cost_function(predicted_y, desired_y):
    loss_function = tf.square(predicted_y - desired_y)
    return tf.reduce_mean(loss_function)

learning_rate = 1.5
optimizer = tf.keras.optimizers.SGD(learning_rate)

def train_step(model, inputs, outputs):
    with tf.GradientTape() as t:
        current_cost_function = cost_function(model(inputs), outputs)
        
    grads = t.gradient(current_cost_function, [model.mu, model.sig])
    optimizer.apply_gradients(zip(grads,[model.mu, model.sig]))
    return current_cost_function

model = Model()

training_epochs = 50
for epoch in range(training_epochs):
    for i in range(0, len(X_train)):
        _cost_function = train_step(model, X_train[i], nY_train[i])        
    if epoch % 10 == 0:
        print("Current cost_function %f" % (_cost_function.numpy()))
     
        
mu_val = model.mu
sig_val = model.sig
print(mu_val.numpy())
print(sig_val.numpy())

plt.scatter(X_train, Y_train)
trY2 = maxY * (np.exp(-np.power(X_train - mu_val, 2.) / (2 * np.power(sig_val, 2.))))
plt.plot(X_train, trY2, 'r')
plt.show()
print("Prediction of week 35", trY2[33])
print("Actual week 35", Y_train[33])

