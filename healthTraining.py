from __future__ import print_function

import os
import numpy as np
import tflearn
import json

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with open('config.json') as config_file:    
    config = json.load(config_file)

def str2bool(v):
  return v.lower() in ("yes", "true", "true", "t", "1")    

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv(config["dataFile"], target_column=0,
                        categorical_labels=True, n_classes=2)

# Build neural network
net = tflearn.input_data(shape=[None, config["numOfColumnExceptTargetColumn"]])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path=config["checkpointPath"])
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=config["numOfEpoch"], batch_size=config["batchSize"], show_metric=str2bool(config["showMetric"]))
# Save model when training is complete to a file
model.save(config["modelFileName"])

print("Network trained and saved as ishealthy.tfl!")

