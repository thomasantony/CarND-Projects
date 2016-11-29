import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.misc import imread

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

data_path = './driving-data/'
img_rows, img_cols = 160, 320
def load_data():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    print('Loading driving log ...')

    driving_log = pd.read_csv(data_path+'driving_log.csv', names=columns)
    num_rows = len(driving_log.index)

    train_images = np.zeros((num_rows, img_rows, img_cols, 3))
    train_steering = driving_log.as_matrix(['steering'])

    for index, row in tqdm(driving_log.iterrows(), unit=' rows', total=num_rows):
        fname = os.path.basename(row['center'])
        # Normalized YUV
        train_images[index] = imread(data_path+'IMG/'+fname, False, 'RGB')/255.

    print('Loaded', num_rows, 'rows.')
    return train_images, train_steering

X_train, y_train = load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=0)

model = Sequential()
model.add(Convolution2D(32, 5, 5,border_mode='valid',
                        input_shape=(img_rows, img_cols, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

print('Compiling model ...')
model.compile('adam', 'mse', metrics=['accuracy'])
print('Training model ...')
h = model.fit(X_train, y_train, batch_size=32, nb_epoch=10,
    validation_data=(X_val, y_val), verbose=1, shuffle=True)
#
# from keras.utils.visualize_util import plot
#
# plot(model, to_file='model.png')
