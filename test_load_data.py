"""
This script uses functions from a variety of modules to ...

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan

file: test_load_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_eeg_data
from plot_raw_and_bootstrap_data import plot_raw_data, bootstrap_p_values, extract_epochs, fdr_correction, \
    plot_confidence_intervals_with_significance
from plot_epoch_data import epoch_data
from clean_data import remove_nan_values, separate_test_and_train_data, separate_artifact_trials, \
    separate_trigger_times_by_class, make_finite_filter, filter_data, get_envelope
from frequency_spectrum_data import get_frequency_spectrum, get_power_spectra_epoched, plot_power_spectrum, \
    get_power_spectra_single, get_frequency_spectrum_single, plot_power_spectrum_single
from plot_results import average_around_electrodes_epoched
from predict_classes import get_predictions, plot_confusion_matrix

# Load the data

# Possible subject labels: 'l1b', 'k6b', or 'k3b'
subject_label = 'l1b'

data_dictionary = load_eeg_data(subject=subject_label)

# Extract the relevant data from the dictionary
raw_data = data_dictionary['Signal']  # Raw EEG signal
fs = data_dictionary['Sampling Frequency']  # The sampling frequency
class_labels = data_dictionary['Class Label']  # All the class labels
trigger_times = data_dictionary['Start Times']  # Start time of each trial
is_artifact_trial = data_dictionary['Artifact Trials']  # Truth data of artifact in each trial
class_label = 1  # Change to be a number 1-4

classes = ["left hand", "right hand", "foot", "tongue"]

# Call to plot_raw_data with your choice of class
plot_raw_data(raw_data, fs, subject_label, class_labels, class_label)

# Separate test and train data

# Separate train and test start times
test_trigger_times, train_trigger_times, training_class_labels = separate_test_and_train_data(class_labels,
                                                                                              trigger_times)

# Separate start time by class
separated_trigger_times = separate_trigger_times_by_class(class_labels, trigger_times)

# Filter the data

# Make the filter
'''Placeholder cutoffs until more appropriate filter determined'''
filter_coefficients = make_finite_filter(low_cutoff=0.1, high_cutoff=8, filter_type='hann', filter_order=50, fs=250)

# Clean the data
raw_data_replaced = remove_nan_values(raw_data)
filtered_data = filter_data(raw_data_replaced, b=filter_coefficients)
envelope = get_envelope(filtered_data)

# Epoch the data

# Epoch raw EEG data
eeg_epochs = epoch_data(fs, trigger_times, raw_data)

# Epoch filtered data
filtered_data_epochs = epoch_data(fs, trigger_times, filtered_data.T, epoch_start_time=2,
                                  epoch_end_time=7)  # Filtering changed shape of data, so use transpose for shape (
# samples, channels)

# Epoch the envelope
envelope_epochs = epoch_data(fs, trigger_times, envelope.T, epoch_start_time=2,
                             epoch_end_time=7)  # Filtering changed shape of envelope from raw data, so use transpose
# for shape (samples, channels)

# Separate clean and artifact epochs

clean_epochs, artifact_epochs, clean_class_labels = separate_artifact_trials(envelope_epochs, is_artifact_trial,
                                                                             class_labels)

# Average around mu and beta electrodes
central_electrodes = [28, 34]
surrounding_map = {
    28: [17, 18, 19, 27, 28, 29, 37, 38, 39],
    34: [23, 24, 25, 33, 34, 35, 43, 44, 45]
}
# class_indices = [1, 4, 5, 6]
class_indices = np.where(class_labels == 3)[0][:3]

average_around_electrodes_epoched(envelope_epochs, central_electrodes, surrounding_map, trials=class_indices,
                                  time=np.arange(2, 7, 1 / fs))

# Frequency spectra of the data

# Take the FFT of the epochs
eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)

# Get the power spectra of each class
spectra_by_class = get_power_spectra_epoched(eeg_epochs_fft, fft_frequencies, class_labels)

# Plot the power spectra
plot_power_spectrum(eeg_epochs_fft, fft_frequencies, spectra_by_class, channels=[28, 31, 34], subject='l1b')

# For filtered data
filtered_epochs_fft, filtered_fft_frequencies = get_frequency_spectrum(filtered_data_epochs, fs)
filtered_spectra_by_class = get_power_spectra_epoched(filtered_epochs_fft, filtered_fft_frequencies, class_labels)
plot_power_spectrum(filtered_epochs_fft, filtered_fft_frequencies, filtered_spectra_by_class, channels=[28, 31, 34],
                    subject='l1b')

# %% Bootstrap for significance

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
plot_confidence_intervals_with_significance(target_erp, rest_erp, erp_times, target_epochs, rest_epochs,
                                            corrected_p_values, subject_label, class_labels, class_label=1,
                                            channels=[0, 1, 2, 3])

# %% Plot by class (fifth class contains test data)
epoch_duration = 1750

# Get epochs by class
class_epochs = []
for class_start_times in separated_trigger_times:
    class_epochs.append(extract_epochs(raw_data, class_start_times, epoch_duration))

# Get ERPs by class
class_erps = []
for class_epoch in class_epochs:
    class_erps.append(np.mean(class_epoch, axis=(0, 2)))
erp_times_classes = np.linspace(0, epoch_duration, num=int(epoch_duration))

# P-values between classes
p_values_classes = []
for class_to_compare_index1 in range(4):  # Only use 4 classes, 5th is test data
    for class_to_compare_index2 in range(4):  # Only use 4 classes, 5th is test data
        # Calculate p-value between different classes (only do one time each)
        if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
            p_values_classes.append(
                bootstrap_p_values(class_epochs[class_to_compare_index1], class_epochs[class_to_compare_index2]))

# Corrected p-values between classes
corrected_p_values_classes = []
for p_value in p_values_classes:
    _, corrected_p_values = fdr_correction(p_value, alpha=0.05)
    corrected_p_values_classes.append(corrected_p_values)

# Plot the classes
# NOTE: As function is written, comparison 1 is "target" and comparison 2 is "rest"
comparison_number = 0
for class_to_compare_index1 in range(4):  # Only use 4 classes, 5th is test data
    for class_to_compare_index2 in range(4):  # Only use 4 classes, 5th is test data
        if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
            plot_confidence_intervals_with_significance(class_erps[class_to_compare_index1],
                                                        class_erps[class_to_compare_index2], erp_times_classes,
                                                        class_epochs[class_to_compare_index1],
                                                        class_epochs[class_to_compare_index2],
                                                        corrected_p_values_classes[comparison_number], subject_label,
                                                        class_labels)

            comparison_number += 1

# Compare frequency data across time in epochs
# import matplotlib as mpl
# mpl.use('TkAgg')
# Get the epochs (rest vs. active MI)
eeg_epochs_rest = epoch_data(fs, trigger_times, raw_data, epoch_start_time=0, epoch_end_time=2)
eeg_epochs_motor_imagery = epoch_data(fs, trigger_times, raw_data, epoch_start_time=3, epoch_end_time=7)

# Get the frequency data of the epochs (resting period)
eeg_epochs_fft_rest, fft_frequencies_rest = get_frequency_spectrum(eeg_epochs_rest, fs)
spectra_by_class = get_power_spectra_epoched(eeg_epochs_fft_rest, fft_frequencies_rest, class_labels)
plot_power_spectrum(eeg_epochs_fft_rest, fft_frequencies_rest, spectra_by_class, channels=[28, 34], subject='l1b')
plt.show()

# Get the frequency data of the epochs (motor imagery)
eeg_epochs_motor_imagery, fft_frequencies_motor_imagery = get_frequency_spectrum(eeg_epochs_motor_imagery, fs)
spectra_by_class = get_power_spectra_epoched(eeg_epochs_motor_imagery, fft_frequencies_motor_imagery, class_labels)
plot_power_spectrum(eeg_epochs_motor_imagery, fft_frequencies_motor_imagery, spectra_by_class, channels=[33, 34],
                    subject='l1b')
plt.show()

# For one epoch
eeg_epochs_motor_imagery = epoch_data(fs, trigger_times, raw_data, epoch_start_time=3, epoch_end_time=7)

index_epoch = 82
current_epoch = eeg_epochs_motor_imagery[index_epoch]

eeg_epoch_fft, fft_frequencies = get_frequency_spectrum_single(current_epoch, fs)
spectrum = get_power_spectra_single(eeg_epoch_fft, fft_frequencies)
# print(spectrum)
print(int(class_labels[index_epoch]))
class_label_current = classes[int(class_labels[index_epoch]) - 1]

plot_power_spectrum_single(fft_frequencies, spectrum, class_label_current, [28], subject='l1b', epoch_index=index_epoch)

# Testing predictions with plot
eeg_epochs_motor_imagery = epoch_data(fs, trigger_times, raw_data, epoch_start_time=3, epoch_end_time=7)

# class 1 with channel 28: 4.5Hz   > 0.
# class 2 with channel 30: 8.75Hz
# class 3 with channel 28: 4Hz     NEGATIVE PEAK < -5
# class 4 with channel 28: 12.75Hz

channel = 28
frequency = 12.5
current_class = 4

class1_indices = np.where(class_labels == current_class)[0]
class1_epoch = eeg_epochs_motor_imagery[class1_indices]

powers = []
for epoch in class1_epoch:
    eeg_epoch_fft, fft_frequencies = get_frequency_spectrum_single(epoch, fs)
    spectrum = get_power_spectra_single(eeg_epoch_fft, fft_frequencies)
    frequency_value = spectrum[channel, np.where(fft_frequencies == frequency)[0][0]]
    powers.append(frequency_value)

print("max: ", np.max(powers))
print("min: ", np.min(powers))
print("mean: ", np.mean(powers))

# predictions
_, train_trigger_times, training_class_labels = separate_test_and_train_data(class_labels, trigger_times)
actual_classes, predicted_classes = get_predictions(raw_data, training_class_labels, train_trigger_times, fs,
                                                    is_artifact_trial, epoch_start_time=3, epoch_end_time=7)
plot_confusion_matrix(actual_classes, predicted_classes, class_names=classes)
