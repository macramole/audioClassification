#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:37:30 2018

@author: leandro
"""

#%% Imports

import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "recordings/"

#%% Get durations

audioFiles = utils.findMusic( DATASET_DIR, "mp3" )
print( "%d files found" % len(audioFiles) )
durations = utils.getDurations( audioFiles )

durations = np.array(durations)
np.save("durations", durations)

#%%

plt.hist ( durations, bins = 100 )
#np.count_nonzero( np.logical_and(durations < 4, durations < 4) ) / len(durations)
np.count_nonzero( durations < 3 ) / len(durations)
sns.kdeplot( durations )
sns

#elijo 3 segundos para truncar