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
import datetime
import time
import pandas as pd

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

audioFiles = []

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
dfAudio = dfAudio.reshape( (dfAudio.shape[0], 130, 20, 1) ) #esto es especifico de 3 segundos

#%% Train

seeds = [ 86143, 93763, 67967 ]
#seeds = [ 86143 ]

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


epochsStep = 50
epochsMax = 300

#lrs = [0.001, 0.01, 0.1, 0.5]
#dropouts = [0.4,0.6,0.8]
#unitss = [ 32, 64, 256, 512]
lrs = [0.00001]
dropouts = [0.4]
unitss = [ 32, 64 ]

for units in unitss:
    for lr in lrs:
        for dropout in dropouts:
            aucs = []
            aucsTrain = []
            tiempos = []
            for idxSplit in range(len(dfSplit)):
                
                print("\n Training seed",idxSplit,"\n")
                
                split = dfSplit[idxSplit]
                X_train, X_test, y_train, y_test = split["X_train"],split["X_test"],split["y_train"],split["y_test"]
                
                ###############################
                ############ MODEL ############
                ###############################
                
                model = Sequential()
                model.add( Conv2D(units, (3,3), input_shape= (130,20,1) ) )
                model.add( Activation("relu") )
                model.add(Dropout(dropout))
                
                model.add(MaxPooling2D(pool_size=(4, 2)))
                model.add(MaxPooling2D(pool_size=(4, 2)))
                model.add(MaxPooling2D(pool_size=(4, 2)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                
#                model.add(Conv2D( int(units/2) , (3, 3)))
#                model.add(Activation('relu'))
                
                
                model.add(Flatten())
                model.add(Dense(1))
                model.add(Activation('sigmoid'))
                
                adam = Adam(lr=lr)
                model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#                model.summary()
                   
                ##############################
                ##############################
                
                aucs.append([])
                aucsTrain.append([])
                tiempos.append([])
                currentEpochs = 0
                for i in range(int(epochsMax/epochsStep)):
                    tic = time.time()
                    model.fit( X_train, y_train, epochs = epochsStep, batch_size = 32, validation_data=(X_test, y_test) )
                    toc = time.time()
                    
                    y_pred = model.predict(X_test)
                    auc = roc_auc_score(y_test, y_pred)
                    
                    y_pred = model.predict(X_train)
                    aucTrain = roc_auc_score(y_train, y_pred)
                    
                    tiempo = toc - tic #(en segundos)
                    
                    print("AUC Train / Test:", aucTrain, "/", auc, "\n")
                        
                    aucs[idxSplit].append(auc)
                    aucsTrain[idxSplit].append(aucTrain)
                    tiempos[idxSplit].append(tiempo)
                    
                    currentEpochs += epochsStep
                    
            with open("results/results.conv.csv", 'a') as resultFile:
                currentEpochs = 0
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for i in range(int(epochsMax/epochsStep)):
                    tiempoPromedio = 0
                    for j in range(len(seeds)):
                        tiempoPromedio += tiempos[j][i]
                    tiempoPromedio = tiempoPromedio / len(seeds)
                    
                    strAucs = ""
                    strAucsTrain = ""
                    for j in range(len(seeds)):
                        strAucs += "%f," % aucs[j][i]
                        strAucsTrain += "%f," % aucsTrain[j][i]
                    strAucs = strAucs + strAucsTrain
                    strAucs = strAucs[0:-1]                    
                    
                    currentEpochs += epochsStep
                    
                    resultFile.write( "%s,%s,%f,%d,%f,%d,%f,%s\n" % (now,DATASET_NAME,lr,units,dropout,currentEpochs,tiempoPromedio,strAucs) )