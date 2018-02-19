# -*- coding: utf-8 -*-

#%% Imports, consts, global vars

#import os
#import librosa
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_DIR = "recordings/"

audioFiles = []

#%% Durations

#durations = np.loadtxt(DATASET_DIR + "durations.txt")
#plt.hist(durations, cumulative=True)
#
#sns.set_style('whitegrid')
#sns.kdeplot( durations )
#
#np.count_nonzero( durations < 0.5 )

#%% Dataset

from keras.utils import to_categorical

audioFiles = utils.findMusic( DATASET_DIR, "wav" )
np.random.shuffle(audioFiles)

labels = [ int( af[ len(DATASET_DIR)] ) for af in audioFiles ]

matrixAudioData = utils.getAudioData( audioFiles )
#matrixAudioData = utils.scaleByRow(matrixAudioData) hacer esto empeora mucho no se por que

cantTrain = int( np.round( len(audioFiles) * 0.7 ) )
X_train = matrixAudioData[0:cantTrain,]
X_test = matrixAudioData[cantTrain:,]
y_train = to_categorical( labels[0:cantTrain] )
y_test = to_categorical( labels[cantTrain:] )

#%% Check
from librosa.display import specshow

unMFCC = X_train[0,:].reshape( (20, int(X_train[0,:].shape[0] / 20)) )
#unMFCC = matrixAudioDataScaled[0,:].reshape( (20, int(matrixAudioDataScaled[0,:].shape[0] / 20)) )

plt.figure(figsize=(10, 4))
specshow(unMFCC, x_axis="time")
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()


matrixAudioDataReshaped = matrixAudioData.reshape((1500, 26, 20))
matrixAudioDataReshaped[0,:,:].shape

#%% PCA

#matrixAudioDataTransformed = utils.doPCA(matrixAudioData)

#%% Feedforward Model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
#from keras.datasets import mnist

model = Sequential()
model.add( Dense(500, input_dim = X_train.shape[1] ) )
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

epochs = 15000
batch_size = 100

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(X_test, y_test) )

plt.title("Feedforward Model")
plt.plot(history.history["loss"], label="Loss function")
plt.plot(history.history["acc"], label="Training Accuracy")
plt.plot(history.history["val_acc"], label="Test Accuracy")
plt.legend()

scores = model.evaluate( X_test, y_test )
print("Accuracy:", scores[1])

#%% LSTM Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence

matrixAudioDataReshaped = matrixAudioData.reshape((1500, 26, 20))

cantTrain = int( np.round( len(audioFiles) * 0.7 ) )
X_train_RNN = matrixAudioDataReshaped[0:cantTrain,]
X_test_RNN = matrixAudioDataReshaped[cantTrain:,]
y_train_RNN = to_categorical( labels[0:cantTrain] )
y_test_RNN = to_categorical( labels[cantTrain:] )

modelLSTM = Sequential()
modelLSTM.add( LSTM(500, input_shape = (26, 20), dropout=0.4, recurrent_dropout=0.3, return_sequences=True ) )
#modelLSTM.add(Dropout(0.4))
modelLSTM.add( LSTM(300, dropout=0.4) )
#modelLSTM.add(Dropout(0.4))
modelLSTM.add(Dense(10))
modelLSTM.add(Activation('softmax'))

modelLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

modelLSTM.summary()

epochs = 50 #10000
batch_size = 100

modelLSTM.fit(X_train_RNN, y_train_RNN, epochs = epochs, batch_size = batch_size, 
              validation_data=(X_test_RNN, y_test_RNN))

plt.title("LSTM Model")
plt.plot(history.history["loss"], label="Loss function")
plt.plot(history.history["acc"], label="Training Accuracy")
plt.plot(history.history["val_acc"], label="Test Accuracy")
plt.legend()

scores = modelLSTM.evaluate( X_test_RNN, y_test_RNN )
print("Accuracy:", scores[1])

#%% Conv2D

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

matrixAudioDataConv2D = matrixAudioData.reshape((1500, 26, 20, 1))
X_train_Conv2D = matrixAudioDataConv2D[0:cantTrain,]
X_test_Conv2D = matrixAudioDataConv2D[cantTrain:,]
y_train_Conv2D = to_categorical( labels[0:cantTrain] )
y_test_Conv2D = to_categorical( labels[cantTrain:] )

modelConv2D = Sequential()
modelConv2D.add( Conv2D(32, (3,3), input_shape= (26,20,1) ) )
modelConv2D.add( Activation("relu") )
modelConv2D.add(Conv2D(32, (3, 3)))
modelConv2D.add(Activation('relu'))
modelConv2D.add(MaxPooling2D(pool_size=(2, 2)))
modelConv2D.add(Dropout(0.25))

modelConv2D.add(Flatten())
modelConv2D.add(Dense(10))
modelConv2D.add(Activation('softmax'))

modelConv2D.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 200
batch_size = 128

history = modelConv2D.fit(X_train_Conv2D, y_train_Conv2D, epochs = epochs, batch_size = batch_size, 
              validation_data=(X_test_Conv2D, y_test_Conv2D))

plt.title("Conv2D Model")
plt.plot(history.history["loss"], label="Loss function")
plt.plot(history.history["acc"], label="Training Accuracy")
plt.plot(history.history["val_acc"], label="Test Accuracy")
plt.legend()

scores = modelConv2D.evaluate( X_test_Conv2D, y_test_Conv2D )
print("Accuracy:", scores[1])