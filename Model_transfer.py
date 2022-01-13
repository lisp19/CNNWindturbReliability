# This file is used for construct a model on target domain 

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

# Load data from local disk
X = np.load('Data/Test.npy',allow_pickle = True)
np.random.shuffle(X)

# Data spliting and reorganize
X_train,X_test,Y_train,Y_test = train_test_split(X[:,0:75],X[:,75],test_size = 0.25)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
X_test = tf.convert_to_tensor(X_test,dtype = 'float32')
Y_test = tf.convert_to_tensor(to_categorical(Y_test))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_train = tf.convert_to_tensor(X_train,dtype = 'float32')
Y_train = tf.convert_to_tensor(to_categorical(Y_train))

X_test_1 = pd.read_pickle('Data/Test1.pkl')
X_test_0 = pd.read_pickle('Data/Test0.pkl')
testnum = len(X_test_1) + len(X_test_0)

# Callback function to save current best model on validation data
cp = ModelCheckpoint('Model_transfer/Best_Model_Transfer.hdf5', monitor = 'val_accuracy', verbose=1,save_best_only=True,mode='max',period = 1)
callbacks_list = [cp]

# Load model on the source domain from disk
model_src = load_model('Model/Best_Model.hdf5')

# Construct model on the target domain 
# Remove the last 5 layers
# Set front layers untrainable to keep model parameters
model = Sequential()
for layer in model_src.layers[:-5]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

# Adding new layers
model.add(Dense(128,name = 'new1'))
model.add(LeakyReLU(alpha = 0.05, name = 'relu1'))
model.add(Dropout(rate = 0.5, name = 'drop01'))
model.add(Dense(64,name = 'new2'))
model.add(LeakyReLU(alpha = 0.05, name = 'relu2'))
model.add(Dropout(rate = 0.5, name = 'drop02'))
model.add(Dense(32,name = 'new3', activation = 'tanh'))
model.add(Dense(2,activation = 'softmax', name = 'output'))
adm = optimizers.Adam(learning_rate = 0.0001, decay = 1e-6)
model.compile(loss = 'categorical_crossentropy',optimizer = adm,metrics=['accuracy'])
model.summary()

# Visualize the training process
# Can be replaced by tensorboard for deep research
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History'+"-"+train)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    pass

# Model training
train_history = model.fit(x = X_train, y = Y_train,epochs = 300,   verbose = 1, validation_split = 0.1,  callbacks = callbacks_list, shuffle = True)


show_train_history(train_history,'accuracy','val_accuracy') 
show_train_history(train_history,'loss','val_loss')

# Model evaluation
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