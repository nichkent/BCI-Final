# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:34:26 2024

@author: nicho
"""
#%% Load the data

# Import
import loadmat

# Select subject data
# Please change to one of the following: l1b, k6b, or k3b
subject_label = "k3b"

# Find the file path
data_file = f"data/{subject_label}.mat"

# Load the data from the file
data = loadmat.loadmat(data_file)

# Grab relevant fields from the data file
s = data['s'] # EEG signal data array with potential NaN values indicating breaks or data saturation.

fs = data['HDR']['SampleRate'] # Sample rate
class_label = data['HDR']['Classlabel'] # Labels of each class (i.e. tongue, left hand, right hand, feet) and NAN which indicates the trials of the test set
trig = data['HDR']['TRIG'] # Beginning of each trial
artifact_selection = data['HDR']['ArtifactSelection'] # Indicates trials with artifacts which were visually identified

# Debug
#print(data['HDR']['Classlabel'])
#print(data['HDR'].keys())
print(data['s'])