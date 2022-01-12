# 用于测试模型准确率
# 读取保存在本地的H5/HDF5模型并进行预测

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

X_test_1 = pd.read_pickle('Data/Test1.pkl')
X_test_0 = pd.read_pickle('Data/Test0.pkl')
# 读取测试数据，为12号风机已经给出的数据
# X_test_n 表示标签为n的12号风机对应数据
testnum = len(X_test_1) + len(X_test_0)
#测试数据的总长度

model = load_model('Model_transfer/Best_Model_Transfer.hdf5')
# 相对路径， 读取模型文件，H5/HDF5均可

print('Predicting.............')
# 以下内容用于以给定模型预测
# 对于12号风机的一组数据（一个CSV，450行）
# 分别预测每一行对应的标签
# Majority Vote得到整个文件对应的标签
# 计算准确率
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