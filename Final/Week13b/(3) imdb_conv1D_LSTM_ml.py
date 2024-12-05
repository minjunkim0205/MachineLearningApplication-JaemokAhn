from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
# consider only top 10,000 common words
(x_train,y_train), (x_test,y_test) = imdb.load_data(num_words=10000)
# check labels and counts for neg and positive
np.unique(y_train,return_counts=True)
# cut off reviews after 500 words
x_train_500 = sequence.pad_sequences(x_train,maxlen=500)
x_test_500 = sequence.pad_sequences(x_test,maxlen=500)

##############코드를 작성하세요

model = Sequential()
# embedding layer
model.add(layers.Embedding(input_dim=10000,
                           output_dim = 128,
                           input_length=500))
#conv1D and MaxPooling1D layer
model.add(layers.Conv1D (filters=32,
                         kernel_size=7, 
                         activation='relu'))
model.add(layers.MaxPooling1D(pool_size=5))
model.add(layers.LSTM(units = 32,
                      dropout=0.2,
                      recurrent_dropout=0.5,
                      return_sequences=True))
model.add(layers.LSTM(units = 64,
                      dropout = 0.1,
                      recurrent_dropout=0.2))
model.add(layers.Dense(1))
model.compile(optimizer = RMSprop(),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
epoch_num= 8
history = model.fit(x=x_train_500, 
                    y=y_train, 
                    epochs=epoch_num,
                    batch_size=128,
                    validation_split=0.2)








########코드를 작성하세요
plt.plot([i+1 for i in range(epoch_num)],
         history.history['acc'],
         label = 'Training acc.')
plt.plot([i+1 for i in range(epoch_num)],
         history.history['val_acc'],
         label='Validation acc.')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()