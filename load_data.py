"""
This script contains a function to load and assign the public dataset to usable variables.

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan

file: load_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import Packages
import numpy as np
from matplotlib import pyplot as plt
import loadmat

# Load the data


def load_eeg_data(subject):
    # Create a file name given subject input
    data_file = f"data/{subject}.mat"

    # Load the data from the file
    data = loadmat.loadmat(data_file)

    # Extract relevant fields from the data file
    signal = data['s']  # EEG signal data array with potential NaN values indicating breaks or data saturation
    fs = data['HDR']['SampleRate']  # Sampling rate
    class_label = data['HDR'][
        'Classlabel']  # Labels of each class (left hand=1, right hand=2, foot=3, tongue=4) and NAN (trials of the
    # test set)
    trigger_time = data['HDR']['TRIG']  # Start time of each trial
    artifact_selection = data['HDR'][
        'ArtifactSelection']  # Indicates trials with artifacts which were visually identified

    # Create a new dictionary with relevant fields
    data_dictionary = {'Signal': signal,
                       'Sampling Frequency': fs,
                       'Class Label': class_label,
                       'Start Times': trigger_time,
                       'Artifact Trials': artifact_selection}

    return data_dictionary
