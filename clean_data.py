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

#%% Make filter  

"""
Is bandpass the best type of filter for this case?
Could make more general, decide on finite or infinite. Select parameters from there
"""
  
def make_finite_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=250):
    
    # get Nyquist frequency to use in filter
    nyquist_frequency = fs/2
    
    # get filter coefficients
    filter_coefficients = firwin(filter_order+1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency], window=filter_type, pass_zero='bandpass')
    
    return filter_coefficients

#%% Filter data

"""
Determine what data will be passed
Epochs or all raw data? Could easily be applied to all and then run the epochs over it
Take in a if infinite filter chosen
"""

def filter_epochs(eeg_epochs, b, channels):
    
    # Reshape for better access of data
    reshaped_eeg_epochs = np.transpose(eeg_epochs, (2,0,1))
    
    # Variables of epoch data for sizing
    channel_count = len(channels) # Won't be all channels, only inputs
    epoch_count = reshaped_eeg_epochs.shape[1]
    sample_count = reshaped_eeg_epochs.shape[2]
    
    # Preallocate array
    filtered_epochs = np.zeros((channel_count, epoch_count, sample_count))
    
    # Apply filter to all samples for each channel of each epoch
    for channel in channels:
        
        for epoch_index in range(epoch_count):
            
            filtered_epochs[channel, epoch_index] = filtfilt(b=b, a=1, x=reshaped_eeg_epochs[channel, epoch_index, :])
        
    return filtered_epochs

#%% Generate the envelope

"""
How useful would the envelope be? 
Is this what we want to bootstrap, identify significance for classification?
"""

def get_epoch_envelopes(filtered_data):
    
    # Variables of epoch data for sizing
    channel_count = filtered_data.shape[0] # Won't be all channels
    epoch_count = filtered_data.shape[1]
    sample_count = filtered_data.shape[2]
    
    # Preallocate the array
    envelope = np.zeros([channel_count, epoch_count, sample_count])
    
    # Get the envelope for each epoch of each channel
    for channel_index in range(channel_count):
        
        for epoch_index in range(epoch_count):
        
            envelope[channel_index, epoch_index]=np.abs(hilbert(x=filtered_data[channel_index, epoch_index]))

    return envelope

#%% 
'''FOR ALL DATA'''

def filter_data(data, b):
    
    # variables for sizing
    sample_count = data.shape[0] # 1st dimension of EEG is number of samples
    channel_count = data.shape[1] # 2nd dimension of EEG is number of channels
    
    # preallocate array
    filtered_data = np.zeros([channel_count, sample_count])
    
    # apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        
        filtered_data[channel_index,:] = filtfilt(b=b, a=1, x=data.T[channel_index,:]) # Transpose of data is shape (channel_count, sample_count)
    
    return filtered_data

def get_envelope(data, filtered_data):
    
    # variables for sizing
    channel_count = len(filtered_data) # 1st dimension is number of channels
    sample_count = len(filtered_data.T) # 2nd dimension is number of samples
    
    # preallocate the array
    envelope = np.zeros([channel_count, sample_count])
    
    # get the envelope for each channel
    for channel_index in range(channel_count):
        
        envelope[channel_index]=np.abs(hilbert(x=filtered_data[channel_index]))

    return envelope  