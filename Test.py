# This file is used to test a given model

import tensorflow as tf
import tensorflow.keras
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import math

# Load data with given label
X_test_1 = pd.read_pickle('Data/Test1.pkl')
X_test_0 = pd.read_pickle('Data/Test0.pkl')

testnum = len(X_test_1) + len(X_test_0)

# Load model from local disk
model = load_model('Model_transfer/Best_Model_Transfer.hdf5')

# For each data line, give the label predicted by model
# For a file, use majority vote to decide the file label
print('Predicting.............')

acc = 0
for i in range(len(X_test_1)):
    X_test_1[i] = X_test_1[i].reshape(X_test_1[i].shape[0],X_test_1[i].shape[1],1)
    x = tf.convert_to_tensor(X_test_1[i], dtype = 'float32')
    pred = model.predict(x)
    temp = 0
    for j in pred:
        if j[1] > 0.5:
            temp += 1
    if 2 * temp >= len(pred):
        acc += 1

for i in range(len(X_test_0)):
    X_test_0[i] = X_test_0[i].reshape(X_test_0[i].shape[0],X_test_0[i].shape[1],1)
    x = tf.convert_to_tensor(X_test_0[i], dtype = 'float32')
    pred = model.predict(x)
    temp = 0
    for j in pred:
        if j[0] > 0.5:
            temp += 1
    if 2 * temp >= len(pred):
        acc += 1

print('The accuracy of prediction is: {}'.format(float(acc)/testnum))