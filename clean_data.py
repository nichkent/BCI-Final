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
from matplotlib import pyplot as plt
from scipy.signal import firwin, filtfilt, freqz, hilbert

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
"""

#%% Make filter  
  
def make_bandpass_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=250):
    
    # get filter coefficients
    nyquist_frequency = fs/2 # get Nyquist frequency to use in filter
    filter_coefficients = firwin(filter_order+1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency], window=filter_type, pass_zero='bandpass')
    
    return filter_coefficients

#%% Filter data

def filter_data(data, b):
    
    # extract data from the dictionary
    eeg = data['eeg']*(10**6) # convert to microvolts
    
    # variables for sizing
    channel_count = len(eeg) # 1st dimension of EEG is number of channels
    sample_count = len(eeg.T) # 2nd dimension of EEG is number of samples
    
    # preallocate array
    filtered_data = np.zeros([channel_count, sample_count])
    
    # apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        
        filtered_data[channel_index,:] = filtfilt(b=b, a=1, x=eeg[channel_index,:])
    
    return filtered_data

#%% Generate the envelope

def get_envelope(filtered_data):
    
    # variables for sizing
    channel_count = len(filtered_data) # 1st dimension is number of channels
    sample_count = len(filtered_data.T) # 2nd dimension is number of samples
    
    # preallocate the array
    envelope = np.zeros([channel_count, sample_count])
    
    # get the envelope for each channel
    for channel_index in range(channel_count):
        
        envelope[channel_index]=np.abs(hilbert(x=filtered_data[channel_index]))
        

    return envelope  