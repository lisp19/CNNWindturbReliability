# This file is used for the data preprocessing
import numpy as np
import os
import time
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

T1 = time.time()
# Data is the labels of current files
data = pd.read_csv('Source/train_labels.csv')


print(data.info())
print(data.head(20))

# datasets is used to storage the train data
datasets = {}
errors = set()
n = data.shape[0]

# Get the file name and store in dictionary
for i in range(n):
	path = 'Source/' + ('0'+str(data.iloc[n - i - 1,0])if data.iloc[n - i - 1,0]>=10 else '00'+str(data.iloc[n - i - 1,0]))+ '/' + data.iloc[n - i - 1,1]
	if os.path.exists(path):
		if data.iloc[n - i - 1,2] != 1.0 and  data.iloc[n - i - 1,2] != 0.0:
			data.iloc[n - i - 1,2] = 'PREDICTION'
		if (data.iloc[n - i - 1,0],data.iloc[n - i - 1,2]) in datasets.keys():
			if data.iloc[n - i - 1,0] == 12:
				datasets[(data.iloc[n - i - 1,1],data.iloc[n - i - 1,2])].append(pd.read_csv(path))
			datasets[(data.iloc[n - i - 1,0],data.iloc[n - i - 1,2])].append(pd.read_csv(path))
		else:
			datasets[(data.iloc[n - i - 1,0],data.iloc[n - i - 1,2])] = []
			print('Reading number %s and ret is %s' % (data.iloc[n - i - 1,0],data.iloc[n - i - 1,2]))
		pass
	else:
		errors.add(data.iloc[n - i - 1,0])

# Data load status
print(len(datasets))
T2 = time.time()
print('The dataloading needs %s seconds.' % (T2-T1))
for key in datasets.keys():
	print('key: %s length: %s' % (key,len(datasets[key])))
print(errors if len(errors) != 0 else 'No error occurs when loading.')

# Data normalization with sklearn
scaler = MinMaxScaler(feature_range = (0,1))
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

for i in range(len(X_target)):
    X_target[i] = np.array(X_target[i])
    scaler.fit(X_target[i])
for i in range(len(X_test_1)):
    X_test_1[i] = np.array(X_test_1[i])
    scaler.fit(X_test_1[i])
for i in range(len(X_test_0)):
    X_test_0[i] = np.array(X_test_0[i])
    scaler.fit(X_test_0[i])

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
for i in range(len(X_target)):
    X_target[i] = scaler.transform(X_target[i])
for i in range(len(X_test_1)):
    X_test_1[i] = scaler.transform(X_test_1[i])
for i in range(len(X_test_0)):
    X_test_0[i] = scaler.transform(X_test_0[i])

# Data reorganization and saving to local disk
X = np.append(X_train,Y_train,axis = 1)
np.random.shuffle(X)
X1 = np.append(X_test,Y_test, axis = 1)
np.random.shuffle(X1)
np.save('Data/Train.npy',X)
np.save('Data/Test.npy',X1)

with open('Data/Test1.pkl','wb') as file1:
    pickle.dump(X_test_1,file1)
with open('Data/Test0.pkl','wb') as file0:
    pickle.dump(X_test_0,file0)

