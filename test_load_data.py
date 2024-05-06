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
from clean_data import remove_nan_values, separate_test_and_train_data, separate_artifact_trials, separate_by_class, make_finite_filter, filter_data, get_envelope
from frequency_spectrum_data import get_frequency_spectrum, get_power_spectra, plot_power_spectrum
from plot_results import average_around_electrodes_epoched
from plot_topo import plot_topo

#%% Load the data

# Possible subject labels: 'l1b', 'k6b', or 'k3b'
subject_label = 'l1b'

data_dictionary = load_eeg_data(subject=subject_label)

#%% Plot the raw data

# Extract the relevant data from the dictionary
raw_data = data_dictionary['Signal']  # Raw EEG signal
fs = data_dictionary['Sampling Frequency']  # The sampling frequency
class_labels = data_dictionary['Class Label']  # All the class labels
trigger_times = data_dictionary['Start Times']  # Start time of each trial
is_artifact_trial = data_dictionary['Artifact Trials'] # Truth data of artifact in each trial
class_label = 1  # Change to be a number 1-4

# Call to plot_raw_data with your choice of class
plot_raw_data(raw_data, fs, subject_label, class_labels, class_label)

#%% Separate test and train data

# Separate train and test start times
test_trigger_times, train_trigger_times, training_class_labels = separate_test_and_train_data(class_labels, trigger_times)

# Separate start time by class
separated_trigger_times = separate_by_class(class_labels, trigger_times)

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
eeg_epochs = epoch_data(fs, trigger_times, raw_data)
# plot_epoch_data(eeg_epochs, fs)

# Epoch filtered data
filtered_data_epochs = epoch_data(fs, trigger_times, filtered_data.T,  epoch_start_time=0, epoch_end_time=10) # Filtering changed shape of data, so use transpose for shape (samples, channels)

# Epoch the envelope
envelope_epochs = epoch_data(fs, trigger_times, envelope.T, epoch_start_time=0, epoch_end_time=10) # Filtering changed shape of envelope from raw data, so use transpose for shape (samples, channels)

#%% Separate clean and artifact epochs

clean_epochs, artifact_epochs = separate_artifact_trials(envelope_epochs, is_artifact_trial)

#%% Average around mu and beta electrodes
central_electrodes = [28, 34]
surrounding_map = {
    28: [17, 18, 19, 27, 28, 29, 37, 38, 39],
    34: [23, 24, 25, 33, 34, 35, 43, 44, 45]
}
# class_indices = [1, 4, 5, 6]
class_indices = np.where(class_labels == 3)[0][:10]

average_around_electrodes_epoched(envelope_epochs, fs, central_electrodes, surrounding_map, trials=class_indices)

#%% Frequency spectra of the data

# Take the FFT of the epochs
eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)

# Get the power spectra of each class
spectra_by_class = get_power_spectra(eeg_epochs_fft, fft_frequencies, class_labels)

# Plot the power spectra
plot_power_spectrum(eeg_epochs_fft, fft_frequencies, spectra_by_class, channels=[28, 31, 34], subject='l1b')

# For filtered data
filtered_epochs_fft, filtered_fft_frequencies = get_frequency_spectrum(filtered_data_epochs, fs)
filtered_spectra_by_class = get_power_spectra(filtered_epochs_fft, filtered_fft_frequencies, class_labels)
plot_power_spectrum(filtered_epochs_fft, filtered_fft_frequencies, filtered_spectra_by_class, channels=[28, 31, 34], subject='l1b')

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
plot_confidence_intervals_with_significance(target_erp, rest_erp, erp_times, target_epochs, rest_epochs, corrected_p_values, subject_label, class_labels, class_label=1, channels=[0,1,2,3])

#%% Plot by class (fifth class contains test data)
epoch_duration=1750

# Get epochs by class
class_epochs = []
for class_start_times in separated_trigger_times:
    class_epochs.append(extract_epochs(raw_data, class_start_times, epoch_duration))

# Get ERPs by class
class_erps = []
for class_epoch in class_epochs:
    class_erps.append(np.mean(class_epoch, axis=(0,2)))
erp_times_classes = np.linspace(0, epoch_duration, num=int(epoch_duration))

# P-values between classes
p_values_classes = []
for class_to_compare_index1 in range(4): # Only use 4 classes, 5th is test data
    for class_to_compare_index2 in range(4): # Only use 4 classes, 5th is test data
        # Calculate p-value between different classes (only do one time each)
        if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
            p_values_classes.append(bootstrap_p_values(class_epochs[class_to_compare_index1], class_epochs[class_to_compare_index2]))

# Corrected p-values between classes
corrected_p_values_classes = []
for p_value in p_values_classes:
    _, corrected_p_values = fdr_correction(p_value, alpha=0.05)
    corrected_p_values_classes.append(corrected_p_values)

# Plot the classes
# NOTE: As function is written, comparison 1 is "target" and comparison 2 is "rest"
comparison_number = 0
for class_to_compare_index1 in range(4): # Only use 4 classes, 5th is test data
    for class_to_compare_index2 in range(4): # Only use 4 classes, 5th is test data
        if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
            
            plot_confidence_intervals_with_significance(class_erps[class_to_compare_index1], class_erps[class_to_compare_index2], erp_times_classes, class_epochs[class_to_compare_index1], class_epochs[class_to_compare_index2], corrected_p_values_classes[comparison_number], subject_label, class_labels)
            
            comparison_number += 1

#%%
# Next Steps:
# Create Topo maps of the subjects
# Create error matrix from all trials for each time-point 0.0 s ≤ t ≤ 7.0 s .