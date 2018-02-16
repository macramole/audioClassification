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

epochs = 10000
batch_size = 100

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(X_test, y_test) )

plt.title("Feedforward Model 500->Drop(0.4)->300->Drop(0.4)")
plt.plot(history.history["loss"], label="Loss function")
plt.plot(history.history["acc"], label="Training Accuracy")
plt.plot(history.history["val_acc"], label="Test Accuracy")
plt.legend()

#%% LSTM Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence

matrixAudioDataReshaped = matrixAudioData.reshape((1500, 26, 20))

cantTrain = int( np.round( len(audioFiles) * 0.7 ) )
X_train = matrixAudioDataReshaped[0:cantTrain,]
X_test = matrixAudioDataReshaped[cantTrain:,]
y_train = to_categorical( labels[0:cantTrain] )
y_test = to_categorical( labels[cantTrain:] )

modelLSTM = Sequential()
modelLSTM.add( LSTM(500, input_shape = (26, 20) ) )
modelLSTM.add(Dense(10))
modelLSTM.add(Activation('softmax'))

rms = RMSprop()
modelLSTM.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

modelLSTM.summary()

epochs = 40 #10000
batch_size = 100

modelLSTM.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(X_test, y_test))

scores = modelLSTM.evaluate( X_test, y_test )
