# 根据模型需求，调整了数据预处理的方式
# 以‘Data_Predicting.py’文件生产的PKL文件为基础
# 生成训练数据、预测数据、预测数据源数据等
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pickle


datasets = pd.read_pickle('Data/datasets.pkl')
data = pd.read_pickle('Data/datasets_predicting.pkl')
scaler = MinMaxScaler(feature_range = (0,1))
# 使用sklearn提供的标准化方法对数据进行归一化


X_train1 = np.array(pd.concat(datasets[(11,1.0)])).astype('float32')
Y_train1 = (np.array([1.0]*len(X_train1)).reshape(len(X_train1),1).astype('float32'))
X_train0 = np.array(pd.concat(datasets[(11,0.0)])).astype('float32')
Y_train0 = (np.array([0.0]*len(X_train0)).reshape(len(X_train0),1).astype('float32'))
X_train = np.append(X_train1,X_train0,axis = 0).astype('float32')
Y_train = np.append(Y_train1,Y_train0,axis = 0).astype('float32')

X_test1 = np.array(pd.concat(datasets[(12,1.0)])).astype('float32')
Y_test1 = np.array([1.0]*len(X_test1)).reshape(len(X_test1),1).astype('float32')
X_test0 = np.array(pd.concat(datasets[(12,0.0)])).astype('float32')
Y_test0 = np.array([0.0]*len(X_test0)).reshape(len(X_test0),1).astype('float32')
X_test = np.append(X_test1,X_test0,axis = 0).astype('float32')
Y_test = np.append(Y_test1,Y_test0,axis = 0).astype('float32')

X_target = datasets[(12,'PREDICTION')]
X_test_0 = datasets[(12,0)]
X_test_1 = datasets[(12,1)]


for key in datasets.keys():
    if key[0] != 11 and key[0] != 12:
        X_train = np.append(X_train,pd.concat(datasets[key]), axis = 0).astype('float32')
        if key[1] == 1.0:  
            Y_train = np.append(Y_train,np.array([1.0]*len(pd.concat(datasets[key]))).reshape(len(pd.concat(datasets[key])),1).astype('float32'), axis = 0)
            pass
        else:
            Y_train = np.append(Y_train,np.array([0.0]*len(pd.concat(datasets[key]))).reshape(len(pd.concat(datasets[key])),1).astype('float32'), axis = 0)
            pass
        pass
    pass

scaler.fit(X_train)
scaler.fit(X_test)
# 采用X的训练和测试数据对归一化器进行训练
for i in range(len(X_target)):
    X_target[i] = np.array(X_target[i])
    scaler.fit(X_target[i])
for i in range(len(X_test_1)):
    X_test_1[i] = np.array(X_test_1[i])
    scaler.fit(X_test_1[i])
for i in range(len(X_test_0)):
    X_test_0[i] = np.array(X_test_0[i])
    scaler.fit(X_test_0[i])
    pass

for key in data.keys():
    data[key] = np.array(data[key]).astype('float32')
    data[key] = scaler.transform(data[key])
with open('Data/Predicting.pkl','wb') as file:
    pickle.dump(data,file)
