#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:39:23 2024

@author: ClaireLeahy

Sources:
    - Motor Imagery Tasks Based Electroencephalogram Signals Classification Using Data-Driven Features: https://www.sciencedirect.com/science/article/pii/S2772528623000134
"""

# Import packages
import numpy as np
from scipy.signal import firwin, filtfilt, hilbert

#%% Considerations

"""
Motor Imagery Tasks (source)
Consider removing signals outside of 8-30Hz (SMR)
µ (SMR) frequencies: 8-13Hz, sensorimotor cortex
β frequencies: 12-30Hz, motor control, thinking
    If frequency spectrum prominent, indicate non-rest state?
"""

"""
Filtering frequencies will help potentially eliminate noise and ideally clarify relevant EEG signals
Bandpass frequencies can be changed to achieve greatest accuracies
Would it make the most sense to filter then epoch or epoch then filter?
How necessary is the envelope?
"""

"""
Trials (epochs) where data_dictionary['Artifact Trials'] is 1 indicate an artifact has been identified
MUCH less specific process than removing components
Should ICA be performed on those trials (basically assuming electrodes=components) and then remove electrode source from that epoch? 
60 channels present. Observing raw data at pertinent channels may help identify necessity of removing artifact (artifact may have minimal contribution to relevant channels)
Is best way of approaching artifact removal looking at accuracy with and without artifact trials? Don't have a mixing matrix
"""

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
    - raw_data_replaced : <np.array of float>: The updated EEG data with NaN values removed in µV. The size of this array is (samples, channels).

    """
    
    # Copy raw_data to alter later
    raw_data_replaced = raw_data
    
    # Find the median of the raw data (ignoring NaN values)
    channel_count = raw_data.shape[1] # Get the number of channels
    medians = np.zeros(channel_count) # Create empty array to store medians
    for channel_index in range(channel_count): # For each channel
        medians[channel_index] = np.nanmedian(raw_data.T[channel_index]) # Update the array with the median
    
    # Find where the raw data is NaN (data saturation, breaks)
    raw_data_nan = np.isnan(raw_data)
    
    # Find the sample and channel where the raw data is NaN
    is_nan = np.where(raw_data_nan == True)
    nan_sample = is_nan[0] # Contains the samples with NaN
    nan_channel = is_nan[1] # Contains the channel where above sample is NaN
    
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
    - class_labels <np.array of float>: Array containing the class label associated with an epoch, values 1-4 or NaN. The shape is (epochs,).
    - trigger_times <np.array of int>: Array containing the sample index at which and epoch begins. The shape is (epochs,).

    Returns
    -------
    - test_trigger_times : TYPE
        DESCRIPTION.
    - train_trigger_times : TYPE
        DESCRIPTION.
    - training_class_labels : TYPE
        DESCRIPTION.

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
    epoched_data : TYPE
        DESCRIPTION.
    is_artifact_trial : TYPE
        DESCRIPTION.
    class_labels : TYPE
        DESCRIPTION.

    Returns
    -------
    clean_epochs : TYPE
        DESCRIPTION.
    artifact_epochs : TYPE
        DESCRIPTION.
    clean_class_labels : TYPE
        DESCRIPTION.

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

    Parameters
    ----------
    class_labels : TYPE
        DESCRIPTION.
    trigger_times : TYPE
        DESCRIPTION.

    Returns
    -------
    separated_trigger_times : TYPE
        DESCRIPTION.

    """
    
    # Sort class labels
    class1 = np.where(class_labels==1)
    class2 = np.where(class_labels==2)
    class3 = np.where(class_labels==3)
    class4 = np.where(class_labels==4)
    test_class = np.where(np.isnan(class_labels)) # NaN assigned to test data
    
    # Get the start times that correspond to the class
    class1_triggers = trigger_times[class1]
    class2_triggers = trigger_times[class2]
    class3_triggers = trigger_times[class3]
    class4_triggers = trigger_times[class4]
    test_class_triggers = trigger_times[test_class]
    
    separated_trigger_times = [class1_triggers, class2_triggers, class3_triggers, class4_triggers, test_class_triggers]
    
    return separated_trigger_times

#%% Make filter  

"""
Is bandpass the best type of filter for this case?
Could make more general, decide on finite or infinite. Select parameters from there
"""
  
def make_finite_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=250):
    """
    

    Parameters
    ----------
    low_cutoff : TYPE
        DESCRIPTION.
    high_cutoff : TYPE
        DESCRIPTION.
    filter_type : TYPE, optional
        DESCRIPTION. The default is 'hann'.
    filter_order : TYPE, optional
        DESCRIPTION. The default is 10.
    fs : TYPE, optional
        DESCRIPTION. The default is 250.

    Returns
    -------
    filter_coefficients : TYPE
        DESCRIPTION.

    """
    
    # Get Nyquist frequency to use in filter
    nyquist_frequency = fs/2
    
    # Get filter coefficients
    filter_coefficients = firwin(filter_order+1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency], window=filter_type, pass_zero='bandpass')
    
    return filter_coefficients

#%% Filter data

def filter_data(data, b):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    filtered_data : TYPE
        DESCRIPTION.

    """
    
    # Variables for sizing
    sample_count = data.shape[0] # 1st dimension of EEG is number of samples
    channel_count = data.shape[1] # 2nd dimension of EEG is number of channels
    
    # Preallocate array
    filtered_data = np.zeros([channel_count, sample_count])
    
    # Apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        
        filtered_data[channel_index,:] = filtfilt(b=b, a=1, x=data.T[channel_index,:]) # Transpose of data is shape (channel_count, sample_count)
    
    return filtered_data

#%% Generate the envelope of filtered data

def get_envelope(filtered_data):
    """
    

    Parameters
    ----------
    filtered_data : TYPE
        DESCRIPTION.

    Returns
    -------
    envelope : TYPE
        DESCRIPTION.

    """
    
    # Variables for sizing
    channel_count = filtered_data.shape[0] # 1st dimension is number of channels
    sample_count = filtered_data.shape[1] # 2nd dimension is number of samples
    
    # Preallocate the array
    envelope = np.zeros([channel_count, sample_count])
    
    # Get the envelope for each channel
    for channel_index in range(channel_count):
        
        envelope[channel_index]=np.abs(hilbert(x=filtered_data[channel_index]))

    return envelope  
