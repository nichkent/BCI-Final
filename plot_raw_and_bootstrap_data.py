# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:04:27 2024

@author: nicho
"""
#%% Plot the raw data
# Imports
import numpy as np
from matplotlib import pyplot as plt

def plot_raw_data(raw_data, fs, subject_label, class_label, class_labels):
    """
        Plot EEG data for a specific subject and class label.
        
        This function filters EEG data to only include trials of a specified class and plots this subset. Each trial is plotted as a separate line on the graph to allow visualization of differences across trials within the same class. The graph is saved as a PNG file.
        
        Parameters:
            - raw_data <np.array>[TRIALS, DATA_POINTS]: 2D array where each row represents a trial and columns represent data points within that trial.
            - fs <float>: Sampling frequency of the EEG data, used to convert data points to time in seconds.
            - subject_label <str>: Identifier for the subject, used in the title of the plot and to name the output file.
            - class_label <int>: Numeric label of the class to be plotted. Used to filter the trials in the `raw_data`.
            - class_labels <np.array>[TRIALS]: 1D array with the class labels for each trial in `raw_data`, used to identify trials of the specified class.
        
        Effects:
            - Generates and saves a plot of EEG data filtered for a specific class, indicating differences across trials. The plot includes a grid and labels for axes. The figure is saved with a filename indicating the subject and class.
        
        Returns:
            - None
    """

    # Filter data to include only the trials for the selected class
    class_indices = np.where(class_labels == class_label)[0]
    
    # Calculate the time range
    time = np.arange(raw_data.shape[1]) / fs
    
    # Create the size of the figure and the title
    plt.figure(figsize=(12, 6))
    plt.title(f"Class {class_labels} Data for Subject {subject_label}")

    # Include only the class relevant indexes in the graph that's displayed
    for index in class_indices:
        plt.plot(time, raw_data[index], label=f'Trial {index+1}')

    # Create the graph
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.grid()
    plt.tight_layout()
    
    # Save the graph
    filename = f"Class_{class_labels}_Data_Subject_{subject_label}.png"
    plt.savefig(filename)
    
    # Show the graph
    plt.show()

#%% Bootsrap the data
def calculate_se_mean(epochs):
    """
        Calculate Standard Error of the Mean (SEM) for given data.
    
        Parameters:
         - epochs <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS]: 3d array representing an epoch for each event in our data.
             ~ epochs[i][j][k], where i represents the i(th) epoch, 
                                     j represents the j(th) sample in the epoch,
                                     k represents the k(th) channel of data in the epoch.
        Returns:
        - se_mean <np.array>[] : SEM values for each time point across trials.
    """    
    # Use numpy to calculate standard deviation for each time point
    std = np.std(epochs, axis=0)
    
    # Number of trials
    n = epochs.shape[0]
    
    # Calculate the standard error of the mean for confidence intervals
    se_mean = std / np.sqrt(n)
    
    # Return standard error of the epochs
    return se_mean

def plot_confidence_intervals(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs):
    """
        Plots the ERPs on each channel for target and nontarget events.
        Plots confidence intervals as error bars around these ERPs.
        Shows error range for the ERPs.
         
        Params:
         - target_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for target events
         - nontarget_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for non-target events
         - erp_times <np.ndarray>[SAMPLES_PER_EPOCH] : time points relative to flashing onset.
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
             
        Returns:
         - None
    """    
    # Calculate se_mean for both target and nontarget ERPs
    target_se_mean = calculate_se_mean(target_epochs)
    nontarget_se_mean = calculate_se_mean(nontarget_epochs)
    
    # Determine the number of channels from the shape of the data
    num_channels = target_erp.shape[1]
    
    # Define the layout of the subplots. Adusts the number of rows based on the number of channels
    cols = 3
    rows = num_channels // cols + (num_channels % cols > 0)
    
    plt.figure(figsize=(10, rows * 3))
    
    # Iterates through all channels
    for channel_index in range(num_channels):
        # Create a subplot of all the channels for the current subject
        plt.subplot(rows, cols, channel_index + 1)
        
        # Pulls target_erp, nontarget_erp, target_se_mean, and nontarget_se_mean for each channel
        target_erp_channel = target_erp[:, channel_index]
        nontarget_erp_channel = nontarget_erp[:, channel_index]
        target_se_mean_channel = target_se_mean[:, channel_index]
        nontarget_se_mean_channel = nontarget_se_mean[:, channel_index]
        
        # Plot target/nontarget ERPs for the current channel
        plt.plot(erp_times, target_erp_channel, label='Target ERP')        
        plt.plot(erp_times, nontarget_erp_channel, label='Non-Target ERP')
        
        # Fill in 95% confidence interval by adding/subtracting 2*SEM to/from mean ERP.
        plt.fill_between(erp_times, target_erp_channel - 2 * target_se_mean_channel, target_erp_channel + 2 * target_se_mean_channel, alpha=0.2, label='Target +/- 95% CI')
        plt.fill_between(erp_times, nontarget_erp_channel - 2 * nontarget_se_mean_channel, nontarget_erp_channel + 2 * nontarget_se_mean_channel, alpha=0.2, label='Non-Target +/- 95% CI')
        
        plt.xlabel('Time (ms)') # X axis label
        plt.ylabel('Amplitude (µV)') # Y axis label
        plt.title(f'Channel {channel_index}') # Title
        
        if (channel_index == num_channels - 1):
            plt.legend()
    
    # Show plot in a tight layout
    plt.tight_layout()
    #plt.show()


def bootstrap_p_values(target_epochs, nontarget_epochs, num_iterations=3000):
    """
        Calculate p-values for each time point and channel using bootstrapping.
    
        Parameters:
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
         - num_iterations <int> : number of bootstrapping iterations
    
        Returns:
         - p_values: np.array, shape (SAMPLES_PER_EPOCH, NUM_CHANNELS), p-values for each time point and channel
    """
    # Combine target and non-target epochs for resampling under the null hypothesis
    combined_epochs = np.concatenate((target_epochs, nontarget_epochs), axis=0)
    
    # Difference between target and non-target means, for later use
    observed_diff = np.mean(target_epochs, axis=0) - np.mean(nontarget_epochs, axis=0)
    
    # Initialize an array to hold the bootstrap differences
    bootstrap_diffs = np.zeros((num_iterations, observed_diff.shape[0], observed_diff.shape[1]))
    
    # Total number of epochs for resampling
    num_epochs = combined_epochs.shape[0]
    
    for i in range(num_iterations):
        # Sample with replacement from the combined set of epochs
        bootstrap_epoch_indices = np.random.randint(0, num_epochs, size=num_epochs)
        # Create bootstrap samples for target and non-target
        bootstrap_sample = combined_epochs[bootstrap_epoch_indices]
        bootstrap_target_sample = bootstrap_sample[:len(target_epochs)]
        bootstrap_nontarget_sample = bootstrap_sample[len(target_epochs):]

        # Calculate the difference between the means of the bootstrap samples
        bootstrap_diff = np.mean(bootstrap_target_sample, axis=0) - np.mean(bootstrap_nontarget_sample, axis=0)
        bootstrap_diffs[i] = bootstrap_diff
    
    # Calculate proportion of bootstrap differences at least as extreme as the observed difference
    p_values = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff), axis=0) / num_iterations
    
    return p_values

