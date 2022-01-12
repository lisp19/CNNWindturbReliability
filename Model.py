# 模型构建的代码

import tensorflow as tf
import tensorflow.keras
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
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

X = np.load('Data/Train.npy',allow_pickle = True)
np.random.shuffle(X)
Test = np.load('Data/Test.npy',allow_pickle = True)
np.random.shuffle(Test)
# 读取训练和测试数据并打乱

X_train,X_test,Y_train,Y_test = train_test_split(X[:,0:75],X[:,75],test_size = 0.1)
Y_train = Y_train.reshape(len(Y_train),1)
# 采用sklearn自带训练集划分方法，将12号的90%数据划入训练集
# 本部分可以调整，取决于对模型的预期

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
X_test = tf.convert_to_tensor(X_test,dtype = 'float32')
Y_test = tf.convert_to_tensor(to_categorical(Y_test))


X_test_1 = pd.read_pickle('Data/Test1.pkl')
X_test_0 = pd.read_pickle('Data/Test0.pkl')
testnum = len(X_test_1) + len(X_test_0)

count = 1
batch_size = 256
step = math.ceil(len(X)/float(batch_size))
# 全局变量，用于服务生成器等

cp = ModelCheckpoint('Model/Best_Model.hdf5', monitor = 'val_accuracy', verbose=1,save_best_only=True,mode='max',period = 1)
callbacks_list = [cp]
# 回调函数，用于在每一个epoch后比较并保存当前最好的模型

model = Sequential()
conv1 = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', name = 'cov1', input_shape = (75,1))
conv2 = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', name = 'cov2')
pool1 = GlobalMaxPooling1D()
flatten = Flatten()
hidden0 = Dense(512,name = 'hidden0')
Act0 = LeakyReLU(alpha = 0.05)
Drop0 = Dropout(0.25)
hidden1 = Dense(256,name = 'hidden1')
Act1 = LeakyReLU(alpha = 0.05)
Drop1 = Dropout(0.25)
hidden2 = Dense(128,name = 'hidden2')
Act2 = LeakyReLU(alpha = 0.05)
Drop2 = Dropout(0.25)
hidden3 = Dense(64, activation = 'tanh', name = 'hidden3')
output = Dense(2,activation='softmax',name='output')
model.add(conv1)
model.add(conv2)
model.add(pool1)
model.add(flatten)
model.add(hidden0)
model.add(Act0)
model.add(Drop0)
model.add(hidden1)
model.add(Act1)
model.add(Drop1)
model.add(hidden2)
model.add(Act2)
model.add(Drop2)
model.add(hidden3)
model.add(output)
model.summary()
adm = optimizers.Adam(learning_rate = 0.0001, decay = 1e-6)
model.compile(loss = 'categorical_crossentropy',optimizer = adm,metrics=['accuracy'])
# 模型搭建和编译


def generateData(X,batch_size):
    global count
    while True:
        batch_x = X[(count - 1)*batch_size:count * batch_size,0:75]
        batch_x = tf.convert_to_tensor(batch_x.reshape(batch_x.shape[0],batch_x.shape[1],1),dtype = 'float32')
        batch_y = tf.convert_to_tensor(to_categorical(X[(count - 1)*batch_size:count * batch_size,75]))
        count = count + 1
        if count * batch_size > len(X):
            count = 1
            np.random.shuffle(X)
        yield(batch_x,batch_y)
        pass
    pass
# 生成器， 用于每次生成一个batch的数据用于训练
# 生成器的实现可优化
def show_train_history(train_history,train,validation):#建立显示训练过程，类似于tensorbroad
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History'+"-"+train)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    pass
# 从助教给的代码里面抄的可视化代码


train_history = model.fit(x = generateData(X,batch_size
    ),epochs = 150,   verbose = 1, validation_data = (X_test,Y_test), steps_per_epoch = step, callbacks = callbacks_list, shuffle = True)
model.save('Model/Model.h5')
# 训练完成后保存一遍
# 实际运行时约在30个epoch左右收敛，实际操作中当验证准确率超过99%即可停止，此时采用已经保存的最优模型作为源领域模型
show_train_history(train_history,'accuracy','val_accuracy') 
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(X_test,Y_test)

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




