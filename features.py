# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of PPG data
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
import math
from scipy.stats import entropy
#from scipy.signal import find_peaks

def _compute_mean_features(window):
    return np.mean(window, axis=0)


# TODO: define functions to compute more features

def _compute_standard_dev_features(window):
    return np.std(window, axis=0)

#def _compute_dom_freq(window):
    #return np.fft.rfft(window, axis=0).astype(float)[1]

def _compute_max(window):
    return np.max(window, axis=0)

def _compute_min(window):
    return np.min(window, axis=0)

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    #x.append(_compute_mean_features(window))
    #feature_names.append("bpm_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    x.append(_compute_standard_dev_features(window))
    feature_names.append("bpm_std")
    
    #x.append(_compute_dom_freq(window))
    #feature_names.append("bpm_dom")
    
    
    #x.append(_compute_max(window))
    #feature_names.append("bpm_max")
    

    x.append(_compute_min(window))
    feature_names.append("bpm_min")
    

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector