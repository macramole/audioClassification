#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:43 2017

@author: leandro
"""

import librosa
import numpy as np
import os
from math import ceil
import time
import sys

SAMPLE_RATE = 22050
# length of audio to be processed, 
# shorter audio will be 0-padded, 
# longer audio will be truncated
SEGUNDOS_FILA = 0.6 
SIZE_AUDIO_RAW = ceil(SAMPLE_RATE * SEGUNDOS_FILA)

distanceMatrix = None

def setSegundosFila(value):
    global SEGUNDOS_FILA, SIZE_AUDIO_RAW
    
    SEGUNDOS_FILA = value
    SIZE_AUDIO_RAW = ceil(SAMPLE_RATE * SEGUNDOS_FILA)

def findMusic(directory, fileType):
    musicFiles = []

    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/", fileType)
        elif file.endswith(fileType):
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                print("Skipped:", directory + file)

    return musicFiles

def getSTFT(data, superVector = True):
    D = librosa.stft(data)
    D = np.abs(D)

    if superVector:
        return D.reshape( D.shape[0] * D.shape[1] )
    else:
        return D.reshape( (D.shape[1], D.shape[0] ) )
    
def getMelspectrogram(data, superVector = True):
    D = librosa.feature.melspectrogram(data)
#    D = np.abs(D)

    if superVector:
        return D.reshape( D.shape[0] * D.shape[1] )
    else:
#        return D.reshape( (D.shape[1], D.shape[0] ) )
        return D

def getZeroCrossingRate(data):
    zc = librosa.feature.zero_crossing_rate(data)
    return zc.reshape( zc.shape[0] * zc.shape[1] )

def getRMSE(data):
    rmse = librosa.feature.rmse(data)
    return rmse.reshape( rmse.shape[0] * rmse.shape[1] )

def getMFCC(data, superVector = True):
    mfcc = librosa.feature.mfcc(data, sr = SAMPLE_RATE)
    if superVector:
        return mfcc.reshape( mfcc.shape[0] * mfcc.shape[1] )
    else:
        return mfcc.reshape( (mfcc.shape[1], mfcc.shape[0] ) )

def scaleByRow(data):

    scaledData = np.copy(data)

#    if min == None:
#        min = np.min(data)
#
#    if max == None:
#        max = np.max(data)

    #si es una sola matriz
    if len(data.shape) == 2:
        for row in range(0, scaledData.shape[0]):
            min = np.min( scaledData[row,:] )
            max = np.max( scaledData[row,:] )
            scaledData[row, :] = np.divide( ( scaledData[row, :] - min ) , ( max - min) )
    elif len(data.shape) == 3:
        for row in range(0, scaledData.shape[1]):
            min = np.min ( np.min( data[:,row,:], axis = 1 ) )
            max = np.max ( np.max( data[:,row,:], axis = 1 ) )
            scaledData[:, row, :] = np.divide( ( scaledData[:, row, :] - min ) , ( max - min) )


    else:
        scaledData = None


    return scaledData


def unScale(scaledData, min, max):
    return ( scaledData  * ( max - min) ) + min

def getDurations( audioFiles ):
    durations = []
    
    count = 0
    COUNT_NOTICE = 200
    
    for f in audioFiles:
        sys.stdout.write('.')
        sys.stdout.flush()
        
        durations.append( librosa.core.get_duration(filename=f) )
        
        count += 1

        if count % COUNT_NOTICE == 0:
            sys.stdout.write('\n\r')
            print("[", count, "/", len(audioFiles), "]")
            sys.stdout.flush()

    return durations

def getAudioData( audioFiles, superVector = True, features = "mfcc", qtyFilesToProcess = None ):
    count = 0
    countFail = 0
    COUNT_NOTICE = 200
    COUNT_FAIL = 20

    listAudioData = []

    tic = time.clock()

    audioFilesDone = []

    if qtyFilesToProcess == None:
        qtyFilesToProcess = len(audioFiles)

    for i in range(0, qtyFilesToProcess):
        try:
            file = audioFiles[i]
            sys.stdout.write('.')
            sys.stdout.flush()

            tmpAudioData, tmpSampleRate = librosa.core.load(file, sr = SAMPLE_RATE)

            tmpAudioData.resize(SIZE_AUDIO_RAW)

            featuresData = None

            if features == "mfcc":
                featuresData = getMFCC(tmpAudioData, superVector)
            elif features == "stft":
                featuresData = getSTFT(tmpAudioData, superVector)
            elif features == "melspectrogram":
                featuresData = getMelspectrogram(tmpAudioData, superVector)
            else:
                print("Utils:getAudioData - No feature chosen")
                return None

            listAudioData.append( featuresData )
            audioFilesDone.append(file)

            count += 1

            if count % COUNT_NOTICE == 0:
                sys.stdout.write('\n\r')
                print("[", count, "/", qtyFilesToProcess, "]")
                sys.stdout.flush()

        except Exception as ex:
            countFail += 1
            sys.stdout.write('\n\r')
            print(file, "[FAIL]", ex)
            sys.stdout.flush()

            if countFail >= COUNT_FAIL:
                break

    matrixAudioData = np.array(listAudioData, dtype=np.float32)
#    matrixAudioData = matrixAudioData.squeeze(1)
    audioFiles.clear()
    audioFiles += audioFilesDone

    print("")
    print("Matriz final:", matrixAudioData.shape)

    toc = time.clock()
    print("time:", toc - tic)
    return matrixAudioData

def saveAudioData( matrixAudioData, filename ):
    np.save(filename, matrixAudioData)

def loadAudioData( filename ):
    return np.load(filename)

def doPCA( matrixAudioData ):
    from sklearn.decomposition import PCA

    tic = time.clock()

    pca = PCA(n_components=200)
    pca.fit(matrixAudioData)
    print("Variance explained:", pca.explained_variance_ratio_.sum())
    matrixAudioDataTransformed = pca.transform(matrixAudioData)

    toc = time.clock()

    print("shape transformed:", matrixAudioDataTransformed.shape)
    print("time:", toc - tic)
    return matrixAudioDataTransformed

def doHierachicalClustering( matrixAudioDataTransformed, threshold = 0.995 ):
    global distanceMatrix

    from scipy.cluster import hierarchy as h
    from scipy.spatial import distance as dist

    distanceFunction = 'cosine' #canberra, cityblock, braycurtis, euclidean
    linkageType = 'average' #single, complete, weighted, average

    print("Distance function:", distanceFunction)
    print("Linkage type:", linkageType)

    tic = time.clock()

    distanceMatrix = dist.pdist(matrixAudioDataTransformed, distanceFunction)
    clusters = h.linkage(distanceMatrix, linkageType)
    c,d=h.cophenet(clusters, distanceMatrix) #factor cofonético

    toc = time.clock()

    print("Cophenet factor:",c)
    print("time:", toc - tic)

    # THRESHOLD = 0.995
    #THRESHOLD = 0.92
    cutTree = h.cut_tree(clusters, height=threshold)

    return cutTree

def doTSNE( matrixAudioDataTransformed ):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    from scipy.spatial import distance as dist

    tic = time.clock()

    similarities = pairwise_distances( dist.squareform(distanceMatrix), n_jobs = -1)

    tsne = TSNE(n_components=2, metric="precomputed")
    positions = tsne.fit(similarities).embedding_

    toc = time.clock()

    return positions

#matrixAudioData =
#matrixAudioData.shape

#%% functiones para inspeccionar activaciones


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    import keras.backend as K

    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()
