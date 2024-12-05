from keras.datasets import cifar10
from keras import layers,models
from keras.optimizers import Adam
import numpy as np
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

encoder = models.Sequential([
    layers.Input(shape=x_train.shape[1:]),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    ])

y_train_categorical = to_categorical(y_train,10)
y_test_categorical = to_categorical(y_test,10)

encoded_train = encoder.predict(x_train)
encoded_train_flat = encoded_train.reshape(len(encoded_train),-1) # (sample,feature)

encoded_test = encoder.predict(x_test)
encoded_test_flat = encoded_test.reshape(len(encoded_test),-1)

classifier = models.Sequential([
    layers.Input(shape=(encoded_train_flat.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
    ])

learning_rate=0.001
optimizer= Adam(learning_rate=learning_rate)
classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])

history = classifier.fit(encoded_train_flat, y_train_categorical, epochs=30, batch_size=512, 
               validation_data=(encoded_test_flat,y_test_categorical))

predicted_probs = classifier.predict(encoded_test_flat)
predicted_classes = np.argmax(predicted_probs,axis=1)
y_test_1D = y_test.flatten()
output = (y_test_1D == predicted_classes)
print(f'accuracy = {np.mean(output)*100}%')


import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_history_styled(history):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, len(history.history['acc']) + 1), 
                 y=history.history['acc'], label='Training Accuracy', marker='o')
    sns.lineplot(x=range(1, len(history.history['val_acc']) + 1), 
                 y=history.history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy', fontsize=14)
    plt.ylim([0.2,1.0])
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)

    # Loss Plot
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1, len(history.history['loss']) + 1), 
                 y=history.history['loss'], label='Training Loss', marker='o')
    sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), 
                 y=history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss', fontsize=14)
    plt.ylim([0,8])
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.title('CIFAR-10 Encoder-based Classification')
    plt.show()

plot_training_history_styled(history)
