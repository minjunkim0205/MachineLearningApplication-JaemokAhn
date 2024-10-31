# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:10:39 2023

@author: ajm
"""


import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

def read(filename, date_idx, date_parse, year=None, bucket=7):
    days_in_year = 365
    
    freq = {}
    if year != None:
         for period in range(0, int(days_in_year / bucket)):
            freq[period] = 0
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile) #csv 파일을 열고
        next(csvreader)#첫번째 행을 건너뜀(헤더 제거)
        
        for row in csvreader:
            #빈 데이터 확인
            if row[date_idx] == '':#날짜 피드가 빈 경우 건너뜀
                continue
            
            #날짜 파싱
            t = time.strptime(row[date_idx], date_parse)#'%m/%d/%Y %H:%M:%S %p'로 데이터 파싱

            #데이터 필터링 및 집계
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
freq = read('C:/Users/dahae/machine learning/311_call_center.csv', 1, '%m/%d/%Y %H:%M:%S %p', 2014)

X_train = np.asarray(list(freq.keys()))
Y_train = np.asarray(list(freq.values()))
#plt.scatter(X_train,Y_train)

model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),   
    layers.Dense(8, activation='relu'),   
    layers.Dense(1)
])

### start of norm on input data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)#1D->2D
Y_train = scaler.fit_transform(Y_train)#1D->2D

X_train = tf.convert_to_tensor(X_train,dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float32)

learning_rate = 0.1
training_epochs = 1000
optimizer = SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y_train, epochs=training_epochs, batch_size=64)

y_pred = model.predict(X_train)
y_pred=scaler.inverse_transform(y_pred) #원래의 크기로 복구
y_pred[y_pred <0] = 0 #음수값이면 0
X_train = X_train.numpy()*52 #원래의 크기로 복구
Y_train = scaler.inverse_transform(Y_train)

plt.scatter(X_train, Y_train, color='blue', s=5,label='Actual Data')
plt.scatter(X_train, y_pred, color='red', s=5,label='Predicted Data')
plt.legend()
plt.show()

#plt.plot(history.history[ 'loss'])
#print(history.history['loss']) 
#print(history.history['accuracy'])
#### 매개변수 값을 확인하는 방법
parameters = model.get_weights()
######

