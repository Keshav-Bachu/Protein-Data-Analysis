# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:27 2018

@author: Keshav Bachu
"""

import JSONConversion as JC
import ProteinModelTrain as PMT
import numpy as np


trainingModule = JC.loadJsonDatabaseTraining()

Yparams = []
Xparams = []

#loop through and extract the data
for i in range(len(trainingModule)):
    #load in the X and Y params
    Yparams.append(trainingModule[i]['meta']['conditions'])
    Xparams.append(np.asarray(trainingModule[i]['data']['disorder']))
    
    #zero padding to keep inputs the same size
    zero_pad = np.zeros((1, 200 - len(Xparams[i])))
    Xparams[i] = np.append(Xparams[i], zero_pad)
    Xparams[i] = Xparams[i].reshape(200, 1)
    
    #remove all NaNs, and replace with 0 as a representation of not used
    where_are_NaNs = np.isnan(Xparams[i])
    Xparams[i][where_are_NaNs] = 0
    