#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:48:11 2024

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan
"""

#%% Import packages
import numpy as np
from load_data import load_eeg_data
from plot_raw_and_bootstrap_data import plot_raw_data, bootstrap_p_values, extract_epochs, fdr_correction, plot_confidence_intervals_with_significance
from plot_epoch_data import epoch_data
from clean_data import remove_nan_values, separate_artifact_trials, make_finite_filter, filter_data, get_envelope

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
is_artifact_trial = data_dictionary['Artifact Trials'] # Truth data of artifact in each trial
class_label = 1  # Change to be a number 1-4

# Call to plot_raw_data with your choice of class
plot_raw_data(raw_data, fs, subject_label, class_labels, class_label)

#%% Filter the data

# Make the filter
'''Placeholder cutoffs until more appropriate filter determined'''
filter_coefficients = make_finite_filter(low_cutoff=0.1, high_cutoff=8, filter_type='hann', filter_order=50, fs=250)

# Clean the data
raw_data_replaced = remove_nan_values(raw_data)
filtered_data = filter_data(raw_data_replaced, b=filter_coefficients)
envelope = get_envelope(filtered_data)

#%% Epoch the data

# Epoch raw EEG data
eeg_epochs = epoch_data(fs, trigger_time, raw_data)
# plot_epoch_data(eeg_epochs, fs)

# Epoch filtered data
filtered_data_epochs = epoch_data(fs, trigger_time, filtered_data.T) # Filtering changed shape of data, so use transpose for shape (samples, channels)

# Epoch the envelope
envelope_epochs = epoch_data(fs, trigger_time, envelope.T) # Filtering changed shape of envelope from raw data, so use transpose for shape (samples, channels)

#%% Separate clean and artifact epochs

clean_epochs, artifact_epochs = separate_artifact_trials(envelope_epochs, is_artifact_trial)

#%% Bootstrap for significance

epoch_duration = 4 * fs  # Duration of each task epoch, e.g., 4 seconds

fs = data_dictionary['Sampling Frequency']
trigger_times = data_dictionary['Start Times']

# Extract rest periods as non-target data
# Assuming rest periods are the 2 seconds before each trigger
rest_start_times = trigger_times - 2 * fs

# Extract task and rest epochs
target_epochs = extract_epochs(raw_data, trigger_times, epoch_duration)
rest_epochs = extract_epochs(raw_data, rest_start_times, epoch_duration)

# Assuming target_epochs and rest_epochs are defined
# Calculate ERPs
target_erp = np.mean(target_epochs, axis=(0, 2))  # Average across epochs and channels
rest_erp = np.mean(rest_epochs, axis=(0, 2))  # Average across epochs and channels
 # Adjust based on the length of the target or rest ERP data
erp_times = np.linspace(0, epoch_duration, num=target_epochs.shape[1])

# Calculate p-values using bootstrap
p_values = bootstrap_p_values(target_epochs, rest_epochs)

# Adjust p-values for multiple comparisons
_, corrected_p_values = fdr_correction(p_values, alpha=0.05)

# Plot ERPs with confidence intervals and significance markings
plot_confidence_intervals_with_significance(target_erp, rest_erp, erp_times, target_epochs, rest_epochs, corrected_p_values, subject_label)




