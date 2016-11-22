import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 99% after 10 epochs
# model = Sequential()
# model.add(Convolution2D(32, 5, 5,
#                         border_mode='valid',
#                         input_shape=(28, 28, 1)))
# model.add(Activation('relu'))
# # model.add(Convolution2D(32, 3, 3))
# # model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# 99.28%
# model = Sequential()
# model.add(Convolution2D(16, 5, 5,
#                         border_mode='same',
#                         input_shape=(28, 28, 1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# 99.44% @ 15 epoch
model = Sequential()
model.add(Convolution2D(16, 5, 5,
                        border_mode='same',
                        input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

h = model.fit(X_train, Y_train, batch_size=128, nb_epoch=10,
    validation_data=(X_test, Y_test), verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

from keras.utils.visualize_util import plot

plot(model, to_file='model.png')
