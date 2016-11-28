import h5py
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.misc import imread

data_path = './driving-data/'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
img_rows, img_cols = 160, 320

print('Loading driving log ...')

driving_log = pd.read_csv(data_path+'driving_log.csv', names=columns)
num_rows = len(driving_log.index)

train_images = np.zeros((num_rows, img_rows, img_cols, 3))
train_steering = driving_log.as_matrix(['steering'])

for index, row in tqdm(driving_log.iterrows(), unit=' rows', total=num_rows):
    fname = os.path.basename(row['center'])
    # Normalized YUV
    train_images[index] = imread(data_path+'IMG/'+fname, False, 'YCbCr')/255.

print('Loaded', num_rows, 'rows.')
# print('\nSaving to disk ...')
# np.savez_compressed('train_data', train_images=train_images, train_steering=train_steering)
# print('Saved data to train_data.npz.,')
