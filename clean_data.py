"""
This file includes a variety of functions to remove, clean, and/or organize the data. The functions in this file serve to separate test and train data (epochs), artifact and "clean" epochs, NaN data, and offers filtering capabilities.

@author: Claire Leahy

file: clean_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
Sources: - Motor Imagery Tasks Based Electroencephalogram Signals Classification Using Data-Driven Features:
https://www.sciencedirect.com/science/article/pii/S2772528623000134 

Useful abbreviations:
    - EEG: Electroencephalograph
    - MI: Motor imagery
    - fs: Sampling frequency
"""

#%% Import packages
import numpy as np
from scipy.signal import firwin, filtfilt, hilbert


#%% Remove NaN values from raw data

def remove_nan_values(raw_data):
    """
    Description 
    ----------- 
    Function that replaces NaN values in the raw data (points of breaks or data saturation) with the median EEG value for that channel.

    Parameters
    ----------
    - raw_data <np.array of float>: The EEG data in µV. The size of this array is (samples, channels).

    Returns 
    -------
    - raw_data_replaced : <np.array of float>: The updated EEG data with NaN values removed in µV. The size of this array is (sample_count, channel_count).

    """

    # Copy raw_data to alter later
    raw_data_replaced = raw_data

    # Find the median of the raw data (ignoring NaN values)
    channel_count = raw_data.shape[1]  # Get the number of channels
    medians = np.zeros(channel_count)  # Create empty array to store medians
    for channel_index in range(channel_count):  # For each channel
        medians[channel_index] = np.nanmedian(raw_data.T[channel_index])  # Update the array with the median

    # Find where the raw data is NaN (data saturation, breaks)
    raw_data_nan = np.isnan(raw_data)

    # Find the sample and channel where the raw data is NaN
    is_nan = np.where(raw_data_nan == True)
    nan_sample = is_nan[0]  # Contains the samples with NaN
    nan_channel = is_nan[1]  # Contains the channel where above sample is NaN

    # Find number of samples that need to be replaced
    samples_to_remove_count = len(nan_sample)

    # Replace NaN data with median of the channel
    for replace_index in range(samples_to_remove_count):
        raw_data_replaced[nan_sample[replace_index]][nan_channel[replace_index]] = medians[nan_channel[replace_index]]

    return raw_data_replaced


#%% Remove test set trials

def separate_test_and_train_data(class_labels, trigger_times):
    """
    Description
    -----------
    This function separates data into training or testing data based on the presence of a class label.

    Parameters
    ----------
    - class_labels <np.array of float>: Array containing the class label associated with an epoch, values 1-4 or NaN. The shape is (epoch_count,). 
    - trigger_times <np.array of int>: Array containing the sample index at which an epoch begins. The shape is (epoch_count,).

    Returns
    -------
    - test_trigger_times <np.array of int>: Array containing the epoch start times for the testing data trials. The shape is (epoch_count,).
    - train_trigger_times <np.array of int>: Array containing the epoch start times for the training data trials. The shape is (epoch_count,).
    - training_class_labels <np.array of float>: Array containing the MI class labels corresponding to the training epochs.

    """

    # Get epoch indices where data is test data
    is_test = np.isnan(class_labels)

    # Get the start times for test and train sets (as lists)
    test_trigger_times = trigger_times[is_test]
    train_trigger_times = trigger_times[~is_test]

    # Get the class labels of the training data
    training_class_labels = class_labels[~is_test]

    return test_trigger_times, train_trigger_times, training_class_labels


#%% Separate clean epochs from epochs with artifacts

def separate_artifact_trials(epoched_data, is_artifact_trial, class_labels):
    """
    Description
    -----------

    Parameters
    ----------
    - epoched_data <np.array of float>: The EEG data organized by epoch. The shape is (epoch_count, sample_count, channel_count).
    - is_artifact_trial <np.array of bool>: Array containing truth data for whether an epoch contains an artifact as identified by the dataset description. The value is 1 (True) if the trial does contain an artifact. The shape is (epoch_count,).
    - class_labels <np.array of float>: Array containing the class label associated with an epoch, values 1-4 or NaN. The shape is (epoch_count,). 

    Returns
    -------
    - clean_epochs <np.array of float>: The epoched EEG data containing only trials without identified artifacts. The shape is (epoch_count, sample_count, channel_count).
    - artifact_epochs <np.array of float>: The epoched EEG data containing only trials with identified artifacts. The shape is (epoch_count, sample_count, channel_count).
    - clean_class_labels <np.array of int>: The class labels as they correspond to the epochs without identified artifacts. The shape is (epoch_count,).

    """

    # Convert artifact_trials to list for counting
    is_artifact_trial = list(is_artifact_trial)

    # Create empty lists to contain the "clean" and "artifact" epochs (lists because of unknown sizing)
    clean_epochs = []
    artifact_epochs = []

    # Create empty list to contain class labels associated with clean data (lists because of unknown sizing)
    clean_class_labels = []

    # Get epoch count for indexing
    epoch_count = epoched_data.shape[0]

    # Separate clean and artifact epochs
    for epoch_index in range(epoch_count):

        # Updated clean_epochs if there isn't an artifact
        if is_artifact_trial[epoch_index] == 0:
            clean_epochs.append(epoched_data[epoch_index])
            clean_class_labels.append(class_labels[epoch_index])

        # Update artifact_epochs if there is an artifact
        elif is_artifact_trial[epoch_index] == 1:
            artifact_epochs.append(epoched_data[epoch_index])

    # Convert to arrays
    clean_epochs = np.array(clean_epochs)
    artifact_epochs = np.array(artifact_epochs)
    clean_class_labels = np.array(clean_class_labels)

    return clean_epochs, artifact_epochs, clean_class_labels


#%% Separate start times by class

def separate_trigger_times_by_class(class_labels, trigger_times):
    """
    Description
    -----------
    Function to organize the trigger times of the epochs based on class.

    Parameters
    ----------
    - class_labels <np.array of float>: Array containing the class label associated with an epoch, values 1-4 or NaN. The shape is (epoch_count,). 
    - trigger_times <np.array of int>: Array containing the sample index at which an epoch begins. The shape is (epoch_count,).

    Returns
    -------
    - separated_trigger_times <list>: List containing the trigger times that correspond to each MI class. The size is 5 (number of classes), and each element contains an array of size (epoch_count,).

    """

    # Sort class labels
    class1 = np.where(class_labels == 1)
    class2 = np.where(class_labels == 2)
    class3 = np.where(class_labels == 3)
    class4 = np.where(class_labels == 4)
    test_class = np.where(np.isnan(class_labels))  # NaN assigned to test data

    # Get the start times that correspond to the class
    class1_triggers = trigger_times[class1]
    class2_triggers = trigger_times[class2]
    class3_triggers = trigger_times[class3]
    class4_triggers = trigger_times[class4]
    test_class_triggers = trigger_times[test_class]

    separated_trigger_times = [class1_triggers, class2_triggers, class3_triggers, class4_triggers, test_class_triggers]

    return separated_trigger_times


#%% Make filter

def make_finite_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=250):
    """
    Description
    -----------
    Function that generates a finite impulse response filter that may be applied to the EEG data.

    Parameters
    ----------
    - low_cutoff <float>: The lower frequency to be used in the bandpass filter in Hz.
    - high_cutoff <float>: The higher frequency to be used in the bandpass filter in Hz.
    - filter_type <str, optional>: The finite impulse response filter of choice to use in the firwin() function. The default is "hann".
    - filter_order <int, optional>: The order of the filter. The default is 10.
    - fs :<int, optional> The sampling frequency in Hz. The default is 250.

    Returns
    -------
    - filter_coefficients <np.array of float>: Numerator coefficients of the finite impulse response filter, shape (filter_order+1,).

    """

    # Get Nyquist frequency to use in filter
    nyquist_frequency = fs / 2

    # Get filter coefficients
    filter_coefficients = firwin(filter_order + 1, [low_cutoff / nyquist_frequency, high_cutoff / nyquist_frequency],
                                 window=filter_type, pass_zero='bandpass')

    return filter_coefficients


#%% Filter data

def filter_data(data, b):
    """
    Function that applies a finite impulse response filter to the EEG data of interest.

    Parameters
    ----------
    - data <np.array of float>: The data to be filtered. The shape is (sample_count, channel_count)
    - b <np.array of float>: Numerator coefficients of the finite impulse response filter, shape (filter_order+1,).

    Returns
    -------
    - filtered_data <np.array of float>: The EEG data after the filter has been applied. The shape is (sample_count, channel_count).

    """

    # Variables for sizing
    sample_count = data.shape[0]  # 1st dimension of EEG is number of samples
    channel_count = data.shape[1]  # 2nd dimension of EEG is number of channels

    # Preallocate array
    filtered_data = np.zeros([channel_count, sample_count])

    # Apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        filtered_data[channel_index, :] = filtfilt(b=b, a=1, x=data.T[channel_index,:]) # Transpose of data is shape (channel_count, sample_count)

    return filtered_data


#%% Generate the envelope of filtered data

def get_envelope(filtered_data):
    """
    Description
    -----------
    Function that calculates the envelope (magnitude) of the filtered EEG data.

    Parameters
    ----------
    - filtered_data <np.array of float>: The EEG data after the filter has been applied. The shape is (sample_count, channel_count).

    Returns
    -------
    - envelope <np.array of float>: The magnitude (envelope) of the filtered EEG data. The shape is (sample_count, channel_count).

    """

    # Variables for sizing
    channel_count = filtered_data.shape[0]  # 1st dimension is number of channels
    sample_count = filtered_data.shape[1]  # 2nd dimension is number of samples

    # Preallocate the array
    envelope = np.zeros([channel_count, sample_count])

    # Get the envelope for each channel
    for channel_index in range(channel_count):
        envelope[channel_index] = np.abs(hilbert(x=filtered_data[channel_index]))

    return envelope
