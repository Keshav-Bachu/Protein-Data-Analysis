# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:18:57 2018

@author: Keshav Bachu
"""

import json

def loadJsonDatabase():
    jsonFile = open('database.json')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)
    return jsonData

#loads in a 100 examples for testing purposes in the training sets
def loadJsonDatabaseTraining():
    jsonFile = open('database.json')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)[0:7000]
    return jsonData

def loadJsonDatabaseTest():
    jsonFile = open('database.json')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)[7000:]
    return jsonData
    
