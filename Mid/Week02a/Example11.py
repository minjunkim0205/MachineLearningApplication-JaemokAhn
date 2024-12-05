import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

x_train = np.linspace(-1, 1, 1000)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33
model = models.Sequential([layers.Input(shape=(1,)),
                           layers.Dense(1)]) # 특징 백터가 1개다(답이 1개다)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3) # 1000 개중에 30% 데이터만 테스트
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
