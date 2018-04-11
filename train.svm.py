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

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC

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

#%% Train

seeds = [ 86143, 93763, 67967 ]

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


#cs = [ 0.1, 0.01, 2, 4, 10 ]
cs = [ 4, 10, 100 ]
#kernels = ["linear", "poly", "rbf"]
kernels = ["poly"]

n_estimators = [10]

for c in cs:
    for kernel in kernels:
        for n_estimator in n_estimators:
            aucs = []
            tiempos = []
            for idxSplit in range(len(dfSplit)):
                
                print("\n Training seed",idxSplit,"\n")
                
                split = dfSplit[idxSplit]
                X_train, X_test, y_train, y_test = split["X_train"],split["X_test"],split["y_train"],split["y_test"]
                
                ###############################
                ############ MODEL ############
                ###############################
                
                model = BaggingClassifier( SVC( C = c, kernel=kernel, random_state = 666), n_estimators=n_estimator, max_samples=1.0 / n_estimator, n_jobs = -1 )
                   
                ##############################
                ##############################
                
                tic = time.time()
                model.fit( X_train, y_train )
                toc = time.time()
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                tiempo = toc - tic #(en segundos)
                
                print("AUC",auc,"\n")
                    
                aucs.append(auc)
                tiempos.append(tiempo)
                    
            with open("results/results.svm.csv", 'a') as resultFile:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                tiempoPromedio = np.mean(tiempos)
                resultFile.write( "%s,%s,%s,%f,%d,%f,%f,%f,%f\n" % (now,DATASET_NAME,kernel,c,n_estimator,tiempoPromedio,aucs[0],aucs[1],aucs[2]) )
