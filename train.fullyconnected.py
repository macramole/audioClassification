#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:07:12 2018

@author: leandro
"""

#%% Imports, consts, global vars

#import os
#import librosa
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import datetime

audioFiles = []

#%% Dataset mnist-audio

DATASET_DIR = "recordings.numbers/"
DATASET_NAME = "mnist-audio"

audioFiles = utils.findMusic( DATASET_DIR , "wav" )
print("%d audios found" % len(audioFiles))

labels = [ int( af[ len(DATASET_DIR)] ) for af in audioFiles ]
labels = np.array(labels)
labels = list( labels == 4 ) #4 contra el mundo

#dfAudio = utils.getAudioData( audioFiles )
#np.save( "audioData.mnist", dfAudio )

dfAudio = np.load("audioData.npy")

#%% Dataset england-vs-all

DATASET_DIR = "recordings/"
DATASET_NAME = "england-vs-all"

df = pd.read_csv("dataset.csv")
labels = list(df["accent"] == "england")

dfFileNames = list(df.filename.str.replace("cv-valid-train/", DATASET_DIR))

utils.setSegundosFila(3)
#dfAudio = utils.getAudioData( dfFileNames )

#np.save( "audioData", dfAudio )
dfAudio = np.load("audioData.npy")

#%% Train

seeds = [ 86143, 93763, 67967 ]
#seeds = [ 86143, 93763 ]

neuronsLayer1 = [ 500, 1000, 5000, 10000 ]
#neuronsLayer1 = [ 3000, 5000, 10000 ]
neuronsLayer2 = [ 300, 500, 1000, 2000 ]
#neuronsLayer2 = [ 1000 ]
neuronsLayer3 = [ 0, 100, 300 ]

dropOuts = [ 0.2, 0.4, 0.1 ]
#dropOuts = [ 0.9 ]
#epochsStep = 5000
#epochsMax = 30000
epochsStep = 20
epochsMax = 80
batch_sizes = [ 512, 1024, 2048 ]

dfSplit = []

for i in range(0, len(seeds)):
    X_train, X_test, y_train, y_test = train_test_split(dfAudio, labels, random_state = seeds[i])
    
#    y_train = to_categorical( y_train )
#    y_test = to_categorical( y_test )
    y_train = np.array( y_train, dtype=int )
    y_test = np.array( y_test, dtype=int )
    
    dfSplit.append( {
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test 
    } )


cantTests = 0

for n1 in neuronsLayer1:
    for n2 in neuronsLayer2:
        for n3 in neuronsLayer3:
            for dO in dropOuts:
                for bs in batch_sizes:
                    
#                    accuracies = []
                    aucs = []
                    tiempos = []
                    
#                    for split in dfSplit:
                    for idxSplit in range(len(dfSplit)):
                        
                        print("\n Training seed",idxSplit,"\n")
                        
                        split = dfSplit[idxSplit]
                        X_train, X_test, y_train, y_test = split["X_train"],split["X_test"],split["y_train"],split["y_test"]
                        
                        ###############################
                        ############ MODEL ############
                        ###############################
                        model = Sequential()
                        
                        model.add( Dense(n1, input_dim = X_train.shape[1] ) )
                        model.add(Activation('relu'))
                        model.add(Dropout(dO))
                        
                        model.add(Dense(n2))
                        model.add(Activation('relu'))
                        model.add(Dropout(dO))
                        
                        if ( n3 > 0 ):
                            model.add(Dense(n3))
                            model.add(Activation('relu'))
                            model.add(Dropout(dO))
                        
                        model.add(Dense(1))
                        model.add(Activation('sigmoid') )

#                        rms = RMSprop()
                        adam = Adam(lr=0.001)
                        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])                        
                        
                        ##############################
                        ##############################
                        
                        
#                        accuracies[idxSplit] = []
#                        accuracies.append([])
                        aucs.append([])
                        tiempos.append([])
                        currentEpochs = 0
                        for i in range(int(epochsMax/epochsStep)):
                            tic = time.clock()
                            history = model.fit(X_train, y_train, epochs = epochsStep, batch_size = bs, validation_data=(X_test, y_test) )
                            toc = time.clock()
                            
                            currentEpochs += epochsStep
                            
                            tiempo = toc - tic
                            
                            y_pred = model.predict_proba(X_test)
                            auc = roc_auc_score(y_test, y_pred)
                            aucs[idxSplit].append( auc )
                            
                            print("\n AUC",auc,"\n")
                            
                            tiempos[idxSplit].append(tiempo)
                            
#                            accuracy = model.evaluate( X_test, y_test )
#                            accuracy = accuracy[1]
#                            accuracies[idxSplit].append(accuracy)
                            
                            cantTests+=1
                    
                    with open("results/results.fullyconnected.csv", 'a') as resultFile:
                        currentEpochs = 0
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for i in range(int(epochsMax/epochsStep)):
                            tiempoPromedio = (tiempos[0][i] + tiempos[1][i] + tiempos[2][i])/3
                            currentEpochs += epochsStep
                            resultFile.write( "%s,%s,%d,%d,%d,%f,%s,%d,%d,%f,%f,%f,%f\n" % (now,DATASET_NAME,n1,n2,n3,dO,"adam",bs,currentEpochs,tiempoPromedio,aucs[0][i],aucs[1][i],aucs[2][i]) )
