#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:17:22 2024

@author: Arthur
"""

#%% Import packages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from plot_epoch_data import epoch_data
from clean_data import separate_test_and_train_data, separate_artifact_trials
from frequency_spectrum_data import get_power_spectra_single, get_frequency_spectrum_single

#%% Function to get predictions

def get_predictions(raw_data, class_labels, trigger_times, fs, is_artifact_trial, epoch_start_time=3, epoch_end_time=7):
    
    eeg_epochs = epoch_data(fs, trigger_times, raw_data, epoch_start_time, epoch_end_time)
    clean_epochs, _, clean_class_labels = separate_artifact_trials(eeg_epochs, is_artifact_trial, class_labels)
    
    predicted_classes = []
    for i, epoch in enumerate(clean_epochs):
        eeg_epoch_fft, fft_frequencies = get_frequency_spectrum_single(epoch, fs)
        spectrum = get_power_spectra_single(eeg_epoch_fft, fft_frequencies)
    
        # skip when not class
        if np.isnan(class_labels[i]):
            predicted_classes.append('test data')
            continue
    
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
    
    correct_predictions = [0, 0, 0, 0]
    incorrect_predictions = [0, 0, 0, 0]
    for i, actual_class in enumerate(clean_class_labels):
        if np.isnan(actual_class):
            continue
    
        if predicted_classes[i] == actual_class:
            correct_predictions[int(actual_class) - 1] += 1
        else:
            incorrect_predictions[int(actual_class) - 1] += 1
    
    print("Correct: ", correct_predictions)
    print("Incorrect: ", incorrect_predictions)
    print("Accuracy by class: ")
    for i in range(4):
        print(f"Class {i+1}: {(correct_predictions[i]/(correct_predictions[i]+incorrect_predictions[i]))*100:.2f}%")

    return clean_class_labels, predicted_classes
def plot_confusion_matrix(actual_classes, predicted_classes, class_names):
    # Generate the confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Classes')
    plt.ylabel('Actual Classes')
    plt.title('Confusion Matrix')
    plt.show()