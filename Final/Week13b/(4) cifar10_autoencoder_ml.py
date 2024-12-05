
from keras import layers, models
from keras.datasets import cifar10
import matplotlib.pyplot as plt

##############코드를 작성하세요

(x_train, ), (x_test, _) = cifar10.load_data() 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
encoder = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation= 'relu', padding='same'),
    layers.MaxPooling2D ( (2, 2), padding='same'),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
])

decoder = models.Sequential([
    layers.Conv2DTranspose(16, (3, 3), activation= 'relu', padding='same', strides=(2, 2)), 
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2)), 
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

autoencoder = models. Sequential ([encoder, decoder])
autoencoder.compile(optimizer=' adam', loss='mse', metrics=['acc'])
autoencoder.fit(x_train, x_train, epochs=10, batch_size=16,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs)

########코드를 작성하세요

def plot_images(original, decoded, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i])
        plt.axis('off')

plot_images(x_test, decoded_imgs)

print(encoder.summary())
print(decoder.summary())
print(encoded_imgs.shape)
print(decoded_imgs.shape)
