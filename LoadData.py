# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:27 2018

@author: Keshav Bachu
"""
import JSONConversion as JC
import ProteinModelTrain as PMT
import numpy as np
import time

"""
trainingModule = JC.loadJsonDatabaseTraining()
InputSize = 200

Yparams = np.zeros((3,1))
Xparams = np.zeros((InputSize,1))

#loop through and extract the data
for i in range(len(trainingModule)):
    #load in and format the X parameters
    #XparamsTemp = np.asarray(trainingModule[i]['data']['disorder'])
    XparamsTemp = np.asarray(trainingModule[i]['data']['propensity'])
    
    #zero padding to keep inputs the same size
    if(len(XparamsTemp) <  InputSize):
        print(len(XparamsTemp))
        zero_pad = np.zeros((1, InputSize - len(XparamsTemp)))
        XparamsTemp = np.append(XparamsTemp, zero_pad)
        XparamsTemp = XparamsTemp.reshape(InputSize, 1)
        
        #remove all NaNs, and replace with 0 as a representation of not used
        where_are_NaNs = np.isnan(XparamsTemp)
        XparamsTemp[where_are_NaNs] = -1
        
        Xparams = np.append(Xparams, XparamsTemp, 1)
        
        
        #Load in and format the Y parameters
        YparamsTemp = (trainingModule[i]['meta']['conditions']['pH'])/14
        YparamsTemp = np.append(YparamsTemp, trainingModule[i]['meta']['conditions']['ionic strength']/5)
        YparamsTemp = np.append(YparamsTemp, trainingModule[i]['meta']['conditions']['temperature']/300)
        YparamsTemp = YparamsTemp.reshape(3,1)
        
        Yparams = np.append(Yparams, YparamsTemp, 1)
        
        '''
        #loads only 1, for testing
        YparamsTemp = np.asarray(trainingModule[i]['meta']['conditions']['ionic strength'])
        YparamsTemp = YparamsTemp.reshape(1,1)
        Yparams = np.append(Yparams, YparamsTemp, 1)
        '''
        
    
Xparams = Xparams[:, 1:]
Yparams = Yparams[:, 1:]
"""

#time used to test runtime of various changes to in calculating cost etc.
start_time = time.time()

#test network
#weights, prediction = PMT.trainModel(Xparams, Yparams, networkShape = [4, 4, 4, 4, 3])

#final network, shape tennative
netShape = [512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 64, 64, 64, 64,128, 128, 256, 256, 128, 128, 128, 128, 32, 32, 32, 32, 3]
weights, prediction = PMT.trainModel(Xparams, Yparams, netShape = netShape, itterations = 5000,  minibatchSize= 1000, learning_rate=0.00001)


print("--- %s seconds ---" % (time.time() - start_time))


def evalRescale(prediction , Yparams = None):
    prediction[0] *= 14
    prediction[1] *= 5
    prediction[2] *= 300
    
    if(Yparams != None):
        Yparams[0] *= 14
        Yparams[1] *= 5
        Yparams[2] *= 300
        
        checkOutput = np.logical_and(np.greater(prediction, Yparams * 0.98), np.less(prediction, Yparams * 1.02))
        
        Yparams[0] /= 14
        Yparams[1] /= 5
        Yparams[2] /= 300
        return checkOutput
    

"""
Reference to storing the numpy weights
In [819]: N
Out[819]: 
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])
In [820]: data={'N':N}
In [821]: np.save('temp.npy',data)
In [822]: data2=np.load('temp.npy')
          then use np.ndarray.tolist to get back dictionary of weights to input in again!
          useful for retraining weights, for a controlled model
In [823]: data2
Out[823]: 
array({'N': array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])}, dtype=object)
    
In [826]: data2[()]['N']
Out[826]: 
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])
    
    to dict is:
        data2 = data2[()]
"""