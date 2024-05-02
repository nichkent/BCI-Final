"""
This script provides a function for creating epoched eeg data.

@author: Aiden Pricer-Coan

file: plot_epoch_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import Statements
import matplotlib.pyplot as plt
import numpy as np


def epoch_data(fs, trigger_time, eeg_data, epoch_start_time=2, epoch_end_time=7):
    """
    Loads data into epoch blocks for further analysis.

    Parameters:
        fs: int - Sampling frequency in samples per second
        trigger_time: np.array (int) - 1d array, start time for each trial
        eeg_data: np.array (float) - 2d array with EEG values for every channel at each time point
        epoch_start_time: (int) - start time offset from start point of each epoch, default when stimulus occurs
        epoch_end_time: (int) - end time offset from start point of each epoch, default when stimulus ends

    Returns: 
        eeg_epochs: np.array (float) - 3d array of size num trials x samples per epoch x num channels,
        epoch for each event in the EEG data
    """
    # Calculate the number of samples in a single epoch
    samples_per_epoch = round(fs * (epoch_end_time - epoch_start_time))

    # Number of epochs
    num_epochs = len(trigger_time)

    # Create array of zeros
    eeg_epochs = np.zeros([num_epochs, samples_per_epoch, eeg_data.shape[1]])

    # Loop through each event
    for event_number, event_start_index in enumerate(trigger_time):
        # Define the epoch start and end indices
        start_index = event_start_index + int(epoch_start_time * fs)
        end_index = event_start_index + int(epoch_end_time * fs)

        # Ensure indices are within bounds
        if start_index < 0 or end_index > len(eeg_data):
            print(f"Trial {event_number}: Start/end index out of bounds")
            continue

        # Get epoch data
        epoch_data = eeg_data[start_index:end_index, :]
        eeg_epochs[event_number, :epoch_data.shape[0], :] = epoch_data

    return eeg_epochs

'''
def plot_epoch_data(epoch_data, fs):
    """
    Plots the epoched EEG data.

    Parameters:
        epoch_data: np.array - 3D array representing epoched EEG data
        fs: int - Sampling frequency in samples per second
    """
    num_epochs, samples_per_epoch, num_channels = epoch_data.shape
    time = np.arange(samples_per_epoch) / fs  # Time axis

    # Plot each epoch
    for epoch_idx in range(num_epochs):
        plt.figure(figsize=(10, 6))
        for channel_idx in range(num_channels):
            plt.plot(time, epoch_data[epoch_idx, :, channel_idx], label=f'Channel {channel_idx+1}')
        plt.title(f'Epoch {epoch_idx+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('EEG Signal')
        plt.legend()
        plt.grid(True)
        plt.show()
'''

