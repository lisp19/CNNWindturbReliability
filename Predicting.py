#用于给出预测结果

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

X = pd.read_pickle('Data/Predicting.pkl')
n = len(X)
res = np.array([])
keys = np.array([])

model = load_model('Model_transfer/Best_Model_Transfer.hdf5')
# 相对路径， 读取模型文件，H5/HDF5均可

print('Predicting.............')

for key in X.keys():
    temp = X[key]
    temp = temp.reshape(temp.shape[0],temp.shape[1],1)
    temp = tf.convert_to_tensor(temp,dtype = 'float32')
    ans = 0
    pred = model.predict(temp)
    for j in pred:
        if j[1] > 0.1:
            ans += 1
            pass
        pass
    if 2 * ans > len(temp):
        res = np.append(res,1.0)
        pass
    else:
        res = np.append(res,0.0)
        pass
    keys = np.append(keys,key)
    pass

keys = keys.reshape(len(keys),1)
res = res.reshape(len(res),1)
result = np.append(keys, res, axis = 1)
result = pd.DataFrame(result, columns = ['file_name','ret'])
print(result)
result.to_csv('Prediction/Result.csv')