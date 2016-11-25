import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10

from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

np.random.seed(1337)  # for reproducibility

n_classes = 10
flat_img_size = 32*32*3
pool_size = (2, 2)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def normalize_color(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = +0.5

    Xmin = 0.0
    Xmax = 255.0

    norm_img = np.empty_like(image_data, dtype=np.float32)
    norm_img = a + (image_data - Xmin)*(b-a)/(Xmax - Xmin)

    return norm_img

X_train = normalize_color(X_train)
X_test = normalize_color(X_test)

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Convolution2D(128, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, input_shape=(flat_img_size,), name='hidden1'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_classes, name='output'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = RMSprop(lr=0.01)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 70% after 10 epochs
history = model.fit(X_train, Y_train, batch_size=64, nb_epoch=10,
                    validation_data=(X_test, Y_test), verbose=1)

# history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
#                     samples_per_epoch=len(X_train), nb_epoch=nb_epoch)
