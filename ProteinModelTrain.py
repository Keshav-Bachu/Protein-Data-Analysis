# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:36:44 2018

@author: Keshav Bachu
"""
#Using the tensorflow archetecture for development
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import numpy as np
from copy import copy


#Creates the placeholders for X and Y
#Placeholders represent what will be placed as an input later on when generating the model, unknown number of examples
#which allows one to evaluate as many examples as needed (minus things like memory limitations etc)
def createPlaceholders(X, Y):
    Xshape = X.shape[0]
    Yshape = Y.shape[0]
    Xplace = tf.placeholder(tf.float32, shape = (Xshape, None))
    Yplace = tf.placeholder(tf.float32, shape = (Yshape, None))  
    
    return Xplace, Yplace

#Used to create the variables of float 32 with a given shape
#uses a network shape to create the required number of variables for the hidden layers
#this should not be confused with the placeholders from X and Y 
def createVariables(networkShape):
    placeholders = {}
    
    for i in range(1, len(networkShape)):
        placeholders['W' + str(i)] = tf.get_variable(name = 'W' + str(i), shape=[networkShape[i], networkShape[i - 1]], initializer=tf.contrib.layers.xavier_initializer())
        placeholders['b' + str(i)] = tf.get_variable(name = 'b' + str(i), shape=[networkShape[i], 1], initializer=tf.zeros_initializer())
    return placeholders

def setVariables(weightsExist):
    placeholders = {}
    
    for key in weightsExist:
        placeholders[key] = tf.get_variable(name = key, initializer=weightsExist[key])
    return placeholders


#Forward propogation using a, relu and sigmoid function on respective layers (relu 1 - n-1 layer and sigmoid for n layer)
def forwardProp(X, placeholders, networkShape):
    #total number of parameters in the network, divided by 2 for the number of layers within it with X being 0
    totalLength = len(placeholders)/2
    totalLength = int(totalLength)
    
    val1 = X
    val2W = placeholders['W' + str(1)]
    val2b = placeholders['b' + str(1)]
    
    pass_Z = tf.matmul(val2W, val1) + val2b
    pass_A = tf.nn.relu(pass_Z)
    
    """
    for i in range (1, totalLength):
        val_W = placeholders['W' + str(i + 1)]
        val_b = placeholders['b' + str(i + 1)]
        
        pass_Z = tf.matmul(val_W, pass_A) + val_b
        pass_A = tf.nn.relu(pass_Z)
    
    """
    #ResNet techneque, must use residual blocks!
    hold_A = pass_A
    value = networkShape[0]
    counter = 0
    for i in range (1, totalLength):
        val_W = placeholders['W' + str(i + 1)]
        val_b = placeholders['b' + str(i + 1)]
        
        pass_Z = tf.matmul(val_W, pass_A) + val_b
        if(value != networkShape[i + 1]):
            pass_A = tf.nn.relu(pass_Z)
            counter = 1;
            value = networkShape[i + 1]
        elif(value == networkShape[i]):
            if(counter == 1):
                pass_A = tf.nn.relu(pass_Z)
                hold_A = pass_A
                counter += 1
                
            elif(counter == 2):
                pass_A = tf.nn.relu(pass_Z)
                counter += 1
             
            #counter is 3 in this case, pass forward the weights                
            else:
                pass_A = tf.nn.relu(pass_Z + hold_A)
                counter = 0;
        
        #prevents values from going too high, usually not needed
        #pass_A = tf.clip_by_value(pass_A, clip_value_min = -1000, clip_value_max = 1000)
        
        #prevents gradieonts from going too low, only works with relu as relu does not have negative numbers
        pass_A = tf.clip_by_value(pass_A, 0.01, 1000)
    
#    """        
    return pass_Z

#Cost function for this sigmoid network
def computeCost(finalZ, Y):
    #logits = tf.transpose(finalZ)
    #labels = tf.transpose(Y)
    
    logits = finalZ
    labels = Y
    
    #cost = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    cost = tf.reduce_mean(tf.squared_difference(logits, labels))
    
    
    #a = tf.pow(tf.log(tf.abs(10 * logits[0, :] - 10 * labels[0, :]) + 10**-8), 2)
    #a = (1 / (2**0.5)) * tf.square(tf.sqrt(tf.abs(logits[0, :])) - tf.sqrt(tf.abs(labels[0, :])))   #Hellinger distance
    #a = tf.squared_difference(logits[1, :], labels[1, :])
    #b = tf.squared_difference(logits[1, :], labels[1, :])
    #c = tf.squared_difference(logits[2, :], labels[2, :])
    #fin = tf.stack([a,b,c])
    
    #cost = tf.reduce_mean(fin)
    
    return cost

#Training the model, X and Y inputs for training and testing NN
#Network shape - dictates the shape of the network; given as a list
#Learning rate - step size of backprop
#iterations -  Number of iterations of NN
#print_cost - controles if cost is printed every 100 iterations
def trainModel(xTest, yTest,networkShape, xDev = None, yDev = None,  learning_rate = 0.00001, itterations = 1500, print_Cost = True, weightsExist = None, minibatchSize = 1):
 
    ops.reset_default_graph()
    costs = []                      #used to graph the costs at the end for a visual overview/analysis
 
    #Need to first create the tensorflow placeholders
    Xlen = xTest.shape[0]   #get the number of rows, aka features for the input data
    networkShape.insert(0, Xlen)
    
    X, Y = createPlaceholders(xTest, yTest)
    if(weightsExist == None):
        placeholders = createVariables(networkShape)
    else:
        placeholders = setVariables(weightsExist)
    
    #define how Z and cost should be calculated
    Zfinal = forwardProp(X, placeholders, networkShape)
    cost = computeCost(Zfinal, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
    numExamples = xTest.shape[1]
    minibatchNumber = numExamples / minibatchSize
    #Set global variables and create a session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #temp_cost = 0 
        for itter in range(itterations):
            
            mini_cost_total = 0
            minibatches = random_mini_batches(xTest, yTest, minibatchSize)
            
            
            for minibatch in minibatches:
                (mini_X, mini_Y) = minibatch
                
                _, mini_cost = sess.run([optimizer, cost], feed_dict={X:mini_X, Y: mini_Y})
                mini_cost_total += mini_cost / minibatchNumber
            
            
            #_,temp_cost = sess.run([optimizer, cost], feed_dict={X:xTest, Y: yTest})
            
            if(itter % 100 == 0):
                #print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                print("Current mini-cost after itteration: " + str(itter) + " is: \t" + str(mini_cost_total))
                
                
            if(itter % 5 == 0):
                costs.append(mini_cost_total)
                #costs.append(temp_cost)
            
            
        parameters = sess.run(placeholders)
        Youtput = Zfinal.eval({X: xTest, Y: yTest})
        #tf.eval(Zfinal)
        
        #confidance level of 75% can be adjusted later
        prediction = tf.logical_and(tf.greater(Zfinal, Y * 0.85), tf.less(Zfinal, Y * 1.15))
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest}))
        #print ("Test Accuracy:", accuracy.eval({X: xDev, Y: yDev}))
        plt.plot(costs)
        
        #Ztest = sess.run(Zfinal, feed_dict={X:xTest, Y: yTest})
        #Ztest = Ztest >= 0.5
        #prediction = Ztest - yTest
        #prediction = np.abs(prediction)
        #prediction = np.sum(prediction)/prediction.shape[1]
        #print ("Train Accuracy:", 1 - prediction)
        
        #prediction = tf.equal(Zfinal, Y)
        #accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    
        #print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest}))
    return parameters, Youtput


def predictor(weights, networkShape, xTest, yTest):
    ops.reset_default_graph()
    
    networkShape2 = copy(networkShape)
    Xlen = xTest.shape[0]
    networkShape2.insert(0, Xlen)
    
    X, Y = createPlaceholders(xTest, yTest)
    placeholders = setVariables(weights)
    Zfinal = forwardProp(X, placeholders, networkShape2)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        Youtput = Zfinal.eval({X: xTest, Y: yTest})
        prediction = tf.logical_and(tf.greater(Zfinal, Y * 0.95), tf.less(Zfinal, Y * 1.05))
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        checkVector = prediction.eval({X: xTest, Y: yTest})
        print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest}))
    
    return Youtput, checkVector



#NOTE: CODE RECEIVED FROM COURSERA DEEP LEARNING SPECIALIZATION CODE
def random_mini_batches(X, Y, mini_batch_size = 1):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    