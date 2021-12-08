# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pickle

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'Test_Data.csv'
data = np.genfromtxt(data_file, delimiter=',')
#print(np.isnan(data))
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

#print("Reorienting accelerometer data...")
#sys.stdout.flush()
#reset_vars()
#reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
#reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
#data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 10
step_size = 10

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 60
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 60
sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["nostress", "stress"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
#print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Test & Evaluate Classifier
#
# -----------------------------------------------------------------------------
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

for train_index, test_index in cv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
#pickled_model = pickle.load(open('classifier.pickle', 'rb'))
#pickled_model.predict(X_test)

with open('classifierallfeatures.pickle','rb') as modelFile:
     model = pickle.load(modelFile)

#For second classifier only heart rate variability

#with open('classifierhrvariability.pickle','rb') as modelFile:
     #model = pickle.load(modelFile)
        
prediction = model.predict(X)
#score = model.score(X_test, y_test)
np.set_printoptions(threshold=np.inf)
print(prediction)
#print(score)