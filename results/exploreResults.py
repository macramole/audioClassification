#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:12:07 2018

@author: leandro
"""

#%%

import pandas as pd
import numpy as np

df = pd.read_csv("results/results.conv.csv")
df

df["aucMean"] = np.mean( df[ ["auc1","auc2","auc3"] ], axis = 1 )
df["aucTrainMean"] = np.mean( df[ ["aucTrain1","aucTrain2","aucTrain3"] ], axis = 1 )

#%% grafico - imports

import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

from plotly.offline import plot

#%%

data = [
    go.Scatter( x = df["aucMean"], y = df["units"], mode = "markers" )
]
plot(data)


#%% modificacion

df["secondLayer"] = df["secondLayer"].astype(int) + 1
df = df.rename( columns = { "secondLayer" : "# layers" } )

df.to_csv("results/results.conv.csv", index = False)


