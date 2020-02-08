import numpy as np
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Reshape,Flatten,Input,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Activation,BatchNormalization,GlobalAveragePooling2D,Dense
import numpy as np
from keras.callbacks import ReduceLROnPlateau

encoding_dim = 128
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format


'''train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
del train 


X_train = np.array(X_train).astype(np.float32) / 255.0
Y_train = np.array(Y_train)
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
X_test = np.array(X_test).astype(np.float32) / 255.0
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


valLen = int(len(Y_train) * 0.2)
X_train = X_train[valLen:]
Y_train = Y_train[valLen:]

X_val = X_train[:valLen]
Y_val = Y_train[:valLen]
'''
# Loads the training and test data sets (ignoring class labels)
(X_train, _), (X_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
max_value = float(X_train.max())
X_train = X_train.astype('float32') / max_value
X_test = X_test.astype('float32') / max_value
X_train.shape, X_test.shape
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

(X_train.shape, X_test.shape)
X_train = X_train.reshape((len(X_train), 28, 28, 1))
X_test = X_test.reshape((len(X_test), 28, 28, 1))
autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((4, 4, 8)))

# Decoder Layers
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()


autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=["acc"])




reduceLRCallback = ReduceLROnPlateau(factor=0.1,patience=5)
hist = autoencoder.fit(X_train, X_train,epochs=(int)(input("Enter Number of Epochs:		")),batch_size=128,validation_data=(X_test, X_test),callbacks=[reduceLRCallback])

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

ax[0].plot(hist.history["loss"],label="train_loss")
ax[0].plot(hist.history["val_loss"],label="val_loss")
ax[0].legend(loc="upper left")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss value")

ax[1].plot(hist.history["acc"],label="train_acc")
ax[1].plot(hist.history["val_acc"],label="val_acc")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy value")

fig.savefig("Model_Results.png")


encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
encoder.summary()
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)


num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(X_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(X_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(16, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("Digits.png")
plt.show()
