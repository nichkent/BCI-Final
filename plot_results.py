"""
This script provides functions for predicting classes and plot analysis results from the eeg data.

@author: Arthur Dolimier

file: predict_classes.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from plot_epoch_data import epoch_data
from clean_data import separate_test_and_train_data, separate_artifact_trials
from frequency_spectrum_data import get_power_spectra_single, get_frequency_spectrum_single


def get_predictions(raw_data, class_labels, trigger_times, fs, is_artifact_trial, epoch_start_time=3, epoch_end_time=7):
    """
    Processes raw EEG data to predict class labels for each trial, excluding those with artifacts.

    Parameters:
    - raw_data (np.ndarray): The raw EEG data array.
    - class_labels (list): Labels indicating the class of each trial.
    - trigger_times (list): Times at which triggers (events) occur within the EEG data.
    - fs (int): Sampling frequency of the EEG data in Hz.
    - is_artifact_trial (list): Boolean list where True indicates the trial contains an artifact.
    - epoch_start_time (int, optional): Start time for epoching the data relative to the trigger in seconds. Defaults to 3.
    - epoch_end_time (int, optional): End time for epoching the data relative to the trigger in seconds. Defaults to 7.

    Returns:
    - clean_class_labels (list): The class labels for trials without artifacts.
    - predicted_classes (list): The predicted class labels for the clean trials.
    """

    eeg_epochs = epoch_data(fs, trigger_times, raw_data, epoch_start_time, epoch_end_time)
    clean_epochs, _, clean_class_labels = separate_artifact_trials(eeg_epochs, is_artifact_trial, class_labels)

    predicted_classes = []
    for epoch_index, epoch in enumerate(clean_epochs):
        eeg_epoch_fft, fft_frequencies = get_frequency_spectrum_single(epoch, fs)
        spectrum = get_power_spectra_single(eeg_epoch_fft, fft_frequencies)

        # skip when not class
        if np.isnan(class_labels[epoch_index]):
            predicted_classes.append('test data')
            continue

        # Set of rules to make predictions
        if spectrum[34, np.where(fft_frequencies == 12.75)[0][0]] > -1:
            predicted_classes.append(4)
        elif spectrum[28, np.where(fft_frequencies == 20.5)] < -15:
            predicted_classes.append(1)
        elif spectrum[34, np.where(fft_frequencies == 12.5)] < -9:
            predicted_classes.append(2)
        else:
            # We don't have a good frequency for class 3 so for now
            # whatever is not caught by the other rules will be predicted as class 3
            predicted_classes.append(3)

    # Go through each prediction and check if correct
    correct_predictions = [0, 0, 0, 0]
    incorrect_predictions = [0, 0, 0, 0]
    for class_index, actual_class in enumerate(clean_class_labels):
        if np.isnan(actual_class):
            continue

        if predicted_classes[class_index] == actual_class:
            correct_predictions[int(actual_class) - 1] += 1
        else:
            incorrect_predictions[int(actual_class) - 1] += 1

    # Display accuracies
    print("Correct: ", correct_predictions)
    print("Incorrect: ", incorrect_predictions)
    print("Accuracy by class: ")
    for i in range(4):
        print(
            f"Class {i + 1}: {(correct_predictions[i] / (correct_predictions[i] + incorrect_predictions[i])) * 100:.2f}%")

    return clean_class_labels, predicted_classes


def plot_confusion_matrix(actual_classes, predicted_classes, class_names, subject):
    """
    Plots a confusion matrix to visualize the accuracy of predictions.

    Parameters:
    - actual_classes (list or array-like): The true class labels.
    - predicted_classes (list or array-like): The predicted class labels by the model.
    - class_names (list of str): The names of the classes, which correspond to the unique labels.
    - subject (str): subject label
    """

    # Generate the confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)

    # Plot the confusion matrix
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Classes')
    plt.ylabel('Actual Classes')
    plt.title(f'Confusion Matrix for {subject}')
    plt.show()

    fig.savefig(f"{subject}_confusion_matrix.png")


def average_around_electrodes(eeg_data, fs, central_electrodes, surrounding_map):
    """
    Averages the EEG data around specified central electrodes using their surrounding electrodes.

    Parameters:
    - eeg_data (numpy.ndarray): The EEG data array with dimensions [electrodes, time].
    - fs (int): Sampling rate of the EEG data
    - central_electrodes (list): List of central electrodes to average around.
    - surrounding_map (dict): Dictionary mapping each central electrode to a list of surrounding electrodes.
    """
    num_plots = len(central_electrodes)
    fig, axes = plt.subplots(nrows=num_plots, figsize=(10, num_plots * 3))

    time = np.arange(eeg_data.shape[1]) / fs

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot

    for idx, electrode in enumerate(central_electrodes):
        surrounding_electrodes = surrounding_map[electrode]
        # Subtract 1 to convert to zero-indexed Python array indices if necessary
        surrounding_indices = [e - 1 for e in surrounding_electrodes]
        averaged_signal = np.mean(eeg_data[surrounding_indices, :], axis=0)

        ax = axes[idx]
        ax.plot(time, averaged_signal)
        ax.set_title(f"Electrode {electrode} (averaged around {surrounding_electrodes})")
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def test_predictions(raw_data, fs, trigger_times, class_labels, channel, frequency, current_class, epoch_start_time=3, epoch_end_time=7):
    """
    Analyzes and prints the power spectrum values at a specific frequency and channel
    for epochs corresponding to a specified class in EEG data.

    Parameters:
    - raw_data (np.ndarray): Array containing raw EEG data.
    - fs (int): Sampling frequency of the EEG data in Hz.
    - trigger_times (list): Times at which the EEG events/triggers of interest occur.
    - class_labels (np.ndarray): Array of class labels for each EEG event/trigger.
    - channel (int): The index of the EEG channel to analyze.
    - frequency (float): The target frequency in Hz to analyze within the EEG data.
    - current_class (int): The class label of the epochs to analyze.
    - epoch_start_time (int): start time of the epoch
    - epoch_end_time (int): end time of the epoch
    """

    # Epoch the data
    eeg_epochs_motor_imagery = epoch_data(fs, trigger_times, raw_data, epoch_start_time=epoch_start_time, epoch_end_time=epoch_end_time)

    # Filter epochs for the current class
    class1_indices = np.where(class_labels == current_class)[0]
    class1_epoch = eeg_epochs_motor_imagery[class1_indices]

    powers = []
    # Calculate power values at the specified frequency and channel for each epoch
    for epoch in class1_epoch:
        eeg_epoch_fft, fft_frequencies = get_frequency_spectrum_single(epoch, fs)
        spectrum = get_power_spectra_single(eeg_epoch_fft, fft_frequencies)
        frequency_value = spectrum[channel, np.where(fft_frequencies == frequency)[0][0]]
        powers.append(frequency_value)

    # Output the metrics for the power values at the specified channel and frequency
    print("max: ", np.max(powers))
    print("min: ", np.min(powers))
    print("mean: ", np.mean(powers))


def average_around_electrodes_epoched(eeg_epochs, central_electrodes, surrounding_map, trials, time,
                                      show_overall_average=False):
    """
    Averages the EEG data around specified central electrodes using their surrounding electrodes,
    for episodic data formatted as num_trials x samples_per_epoch x num_channels.

    Parameters:
    - eeg_epochs (numpy.ndarray): The EEG data array with dimensions [num_trials, samples_per_epoch, num_channels].
    - central_electrodes (list): List of central electrodes to average around.
    - surrounding_map (dict): Dictionary mapping each central electrode to a list of surrounding electrodes.
    - trials (list): List of trials indices to plot
    """
    # Setup number of trails for the plots
    trials_count = len(trials)
    plots_count = len(central_electrodes)

    all_averages = []

    fig, axes = plt.subplots(nrows=plots_count, ncols=trials_count, figsize=(trials_count * 3, plots_count * 3))
    if plots_count == 1:
        # Adjust the axes array for single subplot cases
        axes = np.expand_dims(axes, axis=0)

    for electrode_index, electrode in enumerate(central_electrodes):
        surrounding_electrodes = surrounding_map[electrode]
        # Adjust as electrode indices are 1-based
        surrounding_indices = [e - 1 for e in surrounding_electrodes]

        for trial_index, trial in enumerate(trials):
            # plot for each trial
            averaged_signal = np.mean(eeg_epochs[trial, :, surrounding_indices], axis=0)
            all_averages.append(averaged_signal)
            ax = axes[electrode_index, trial_index]
            ax.plot(time, averaged_signal)
            ax.set_title(f"Trial {trial + 1}, Electrode {electrode}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Show overall average if desired
    if show_overall_average:
        plt.plot(time, np.mean(all_averages, axis=0))
        plt.show()

