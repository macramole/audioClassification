#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:50:46 2018

@author: leandro
"""

#%% Imports

import utils
import numpy as np
import time

import plotly.plotly as py
import plotly.graph_objs as go


#import pandas as pd

import matplotlib.pyplot as plt

DATASET_DIR = "recordings/"

#%% Load data

from keras.utils import to_categorical

audioFiles = utils.findMusic( DATASET_DIR, "wav" )
np.random.shuffle(audioFiles)

labels = np.array( [ int( af[ len(DATASET_DIR)] ) for af in audioFiles ] )

matrixAudioData = utils.getAudioData( audioFiles )
matrixAudioDataPCA = utils.doPCA( matrixAudioData )

#clusters = utils.doHierachicalClustering( matrixAudioDataPCA )
#tsnePositions = utils.doTSNE( matrixAudioDataPCA )

#%% T-SNE

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance as dist

tic = time.clock()

tsne = TSNE(n_components=2)
#tsneResult = tsne.fit(matrixAudioDataPCA)
tsneResult = tsne.fit(matrixAudioData)
tsnePositions = tsneResult.embedding_

toc = time.clock()

#%% Plot

def onPick(e):
    print(e.mouseevent.x,e.mouseevent.y)
    pass

fig, ax = plt.subplots()

for i in range(0,10):
    filterByLabel = ( labels == i )
    ax.scatter(tsnePositions[filterByLabel,0], tsnePositions[filterByLabel,1], s = 3, label = i, picker=1 )

ax.legend()

fig.canvas.mpl_connect('pick_event', onPick)


