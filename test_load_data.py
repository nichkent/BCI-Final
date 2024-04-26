#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:48:11 2024

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan
"""

#%% Import packages
from load_data import load_eeg_data
from plot_raw_and_bootstrap_data import plot_raw_data
#%% Load the data

# Possible subject labels: 'l1b', 'k6b', or 'k3b'
subject_label = 'l1b'

data_dictionary = load_eeg_data(subject=subject_label)
#%% Plot the raw data

# Extract the relevant data from the dictionary
raw_data = data_dictionary['Signal'] # Raw EEG signal
fs = data_dictionary['Sampling Frequency'] # The sampling frequency
class_labels = data_dictionary['Class Label'] # All the class labels
class_label = 1 # Change to be a number 1-4

# Call to plot_raw_data with your choice of class
plot_raw_data(raw_data, fs, subject_label, class_labels, class_label)