"""
This script provides a function for measuring the change in power spectrum in the beta rhythm band.

@author: Aiden Pricer-Coan

file: measure_power_change.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import statements
import numpy as np
from frequency_spectrum_data import get_power_spectra_epoched
from plot_raw_and_bootstrap_data import extract_epochs
from load_data import load_eeg_data
import loadmat


def measure_power_change(raw_data, trigger_times, class_labels, fs, epoch_duration=5, baseline_duration=2):
    # Initialize dictionary to store average beta power change for each class
    avg_beta_power_change = {}

    # Define frequency bands
    beta_band = [13, 30]  # Frequency range for beta rhythm (Hz)

    # Extract rest and task epochs
    rest_epochs = extract_epochs(raw_data, trigger_times - baseline_duration * fs, epoch_duration * fs)
    task_epochs = extract_epochs(raw_data, trigger_times, epoch_duration * fs)

    # Calculate power spectra for rest and task epochs
    rest_power_spectra = get_power_spectra_epoched(rest_epochs, fs, class_labels)
    task_power_spectra = get_power_spectra_epoched(task_epochs, fs, class_labels)

    # Convert the power spectra lists to numpy arrays
    rest_power_spectra = np.array(rest_power_spectra)
    task_power_spectra = np.array(task_power_spectra)

    # Make sure the power spectra have the correct shape
    rest_power_spectra = np.moveaxis(rest_power_spectra, 0, -1)  # Move the class dimension to the last axis
    task_power_spectra = np.moveaxis(task_power_spectra, 0, -1)  # Move the class dimension to the last axis

    # Calculate average power in beta band for each epoch
    n_fft = rest_power_spectra.shape[2]
    freq_bins = np.fft.rfftfreq(n_fft, d=1/fs)
    freq_indices = (freq_bins >= beta_band[0]) & (freq_bins <= beta_band[1])
    rest_beta_power = np.nanmean(rest_power_spectra[:, :, freq_indices], axis=1)
    task_beta_power = np.nanmean(task_power_spectra[:, :, freq_indices], axis=1)

    # Calculate power change between task and rest for each epoch
    beta_power_change = task_beta_power - rest_beta_power

    # Separate power change by class
    for class_label in np.unique(class_labels):
        # Find indices corresponding to the current class
        class_indices = np.where(class_labels == class_label)[0]
        # Calculate average power change for the class
        avg_beta_power_change[class_label] = np.nanmean(beta_power_change[class_indices])  # Use np.nanmean to handle empty slices

    return avg_beta_power_change







subject_label = 'l1b'
data_dictionary = load_eeg_data(subject='/Users/aiden/PycharmProjects/BCI-Final/l1b.mat')
trigger_times = data_dictionary['Start Times']
raw_data = data_dictionary['Signal']
class_labels = data_dictionary['Class Label']
fs = data_dictionary['Sampling Frequency']

beta_power_change = measure_power_change(raw_data, trigger_times, class_labels, fs)
print(beta_power_change)

