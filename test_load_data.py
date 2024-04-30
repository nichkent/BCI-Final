#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:48:11 2024

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan
"""

#%% Import packages
from load_data import load_eeg_data
from plot_raw_and_bootstrap_data import plot_raw_data
from plot_epoch_data import epoch_data
from clean_data import make_finite_filter, filter_epochs, get_epoch_envelopes

#%% Load the data

# Possible subject labels: 'l1b', 'k6b', or 'k3b'
subject_label = 'l1b'

data_dictionary = load_eeg_data(subject=subject_label)

#%% Plot the raw data

# Extract the relevant data from the dictionary
raw_data = data_dictionary['Signal']  # Raw EEG signal
fs = data_dictionary['Sampling Frequency']  # The sampling frequency
class_labels = data_dictionary['Class Label']  # All the class labels
trigger_time = data_dictionary['Start Times']  # Start time of each trial
class_label = 1  # Change to be a number 1-4

# Call to plot_raw_data with your choice of class
plot_raw_data(raw_data, fs, subject_label, class_labels, class_label)

#%% Epoch the data

# Epoch EEG data
eeg_epochs = epoch_data(fs, trigger_time, raw_data)

# plot_epoch_data(eeg_epochs, fs)

#%% Filter the data

# Make the filter
'''Placeholder cutoffs until more appropriate filter determined'''
filter_coefficients = make_finite_filter(low_cutoff=0.1, high_cutoff=8, filter_type='hann', filter_order=50, fs=250)

'''Placeholder channels until ready to run all, gives user option to select a few'''
# Filter the epochs
filtered_epochs = filter_epochs(eeg_epochs, b=filter_coefficients, channels=[0,1,2])

# Get the envelope of the data
envelope = get_epoch_envelopes(filtered_epochs)

# Bootstrap the envelope