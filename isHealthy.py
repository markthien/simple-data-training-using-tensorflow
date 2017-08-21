from __future__ import division, print_function, absolute_import

import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy
import argparse
import csv
import json
import math

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with open('config.json') as config_file:    
    config = json.load(config_file)

with open(config["testDataFile"], 'rb') as infile:
    reader = csv.reader(infile)
    reader.next()
    testDataList = list(reader)     

# Build neural network
net = tflearn.input_data(shape=[None, config["numOfColumnExceptTargetColumn"]])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path=config["checkpointPath"])

modelEpochFile = config["checkpointPath"] + "-" + str(int(math.ceil(config["numOfRowData"]/config["batchSize"]) * config["numOfEpoch"]))
model.load(modelEpochFile)

# Predict is healthy chances (class 1 results)
pred = model.predict(testDataList)
print("Female 1 Is Healthy Prediction :", pred[0][1])
print("Female 2 Is Healthy Prediction :", pred[1][1])