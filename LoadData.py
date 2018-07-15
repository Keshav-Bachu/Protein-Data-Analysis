# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:27 2018

@author: Keshav Bachu
"""

import JSONConversion as JC
import ProteinModelTrain as PMT
import numpy as np
from tensorflow.python import debug as tf_debug

"""
trainingModule = JC.loadJsonDatabaseTraining()
InputSize = 530

Yparams = np.zeros((3,1))
Xparams = np.zeros((InputSize,1))

#loop through and extract the data
for i in range(len(trainingModule)):
    #load in and format the X parameters
    XparamsTemp = np.asarray(trainingModule[i]['data']['disorder'])
    
    #zero padding to keep inputs the same size
    print(len(XparamsTemp))
    zero_pad = np.zeros((1, InputSize - len(XparamsTemp)))
    XparamsTemp = np.append(XparamsTemp, zero_pad)
    XparamsTemp = XparamsTemp.reshape(InputSize, 1)
    
    #remove all NaNs, and replace with 0 as a representation of not used
    where_are_NaNs = np.isnan(XparamsTemp)
    XparamsTemp[where_are_NaNs] = 0
    
    Xparams = np.append(Xparams, XparamsTemp, 1)
    
    #Load in and format the Y parameters
    YparamsTemp = (trainingModule[i]['meta']['conditions']['pH'])
    YparamsTemp = np.append(YparamsTemp, trainingModule[i]['meta']['conditions']['ionic strength'])
    YparamsTemp = np.append(YparamsTemp, trainingModule[i]['meta']['conditions']['temperature'])
    YparamsTemp = YparamsTemp.reshape(3,1)
    
    Yparams = np.append(Yparams, YparamsTemp, 1)
    
    
Xparams = Xparams[:, 1:]
Yparams = Yparams[:, 1:]
"""

weights = PMT.trainModel(Xparams, Yparams, networkShape = [4, 4, 3])