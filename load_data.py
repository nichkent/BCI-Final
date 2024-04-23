# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:34:26 2024

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan
"""

#%% Import Packages
import numpy as np
from matplotlib import pyplot as plt
import loadmat

# Select subject data
# Please change to one of the following: l1b, k6b, or k3b
subject_label = "k3b"

# Find the file path
data_file = f"data/{subject_label}.mat"

#%% Load the data

"""
    TODO:
        - Prompt user for new input if file directory doesn't exist (if 'l1b', 'k6b', or 'k3b' aren't entered)?
        - Look more into the classifications (1,2,3,4) and corresponding thoughts
        - Docstrings

"""

def load_eeg_data(subject):
    
    # Create a file name given subject input
    data_file = f"data/{subject}.mat"

    # Load the data from the file
    data = loadmat.loadmat(data_file)
    
    # Extract relevant fields from the data file
    signal = data['s'] # EEG signal data array with potential NaN values indicating breaks or data saturation
    
    fs = data['HDR']['SampleRate'] # Sampling rate
    class_label = data['HDR']['Classlabel'] # Labels of each class (left hand=1, right hand=2, foot=3, tongue=4) and NAN (trials of the test set)
    trigger_time = data['HDR']['TRIG'] # Start time of each trial
    artifact_selection = data['HDR']['ArtifactSelection'] # Indicates trials with artifacts which were visually identified
    
    # Create a new dictionary with relevant fields
    data_dictionary = {'Signal': signal,
                       'Sampling Frequency': fs,
                       'Class Label': class_label,
                       'Start Times': trigger_time,
                       'Artifact Trials': artifact_selection}
    
    # Debug
    #print(data['HDR']['Classlabel'])
    #print(data['HDR'].keys())
    #print(data['s'])

    return data_dictionary