# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:27 2018

@author: Keshav Bachu
"""

import JSONConversion as JC
import ProteinModelTrain as PMT
import numpy as np


trainingModule = JC.loadJsonDatabaseTraining()

Yparams = np.zeros((3,1))
Xparams = np.zeros((200,1))

#loop through and extract the data
for i in range(len(trainingModule)):
    #load in and format the X parameters
    XparamsTemp = np.asarray(trainingModule[i]['data']['disorder'])
    
    #zero padding to keep inputs the same size
    zero_pad = np.zeros((1, 200 - len(XparamsTemp)))
    XparamsTemp = np.append(XparamsTemp, zero_pad)
    XparamsTemp = XparamsTemp.reshape(200, 1)
    
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