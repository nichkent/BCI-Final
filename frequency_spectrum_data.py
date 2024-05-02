#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:27:48 2024

@author: ClaireLeahy
"""

#%% Import packages

import numpy as np
from matplotlib import pyplot as plt

#%% Get the frequency spectrum of the data

def get_frequency_spectrum(eeg_epochs, fs):
    """
    Description
    -----------
    Function that takes the Fourier Transform of the epoched EEG data and provides the corresponding frequencies.

    Parameters
    ----------
    eeg_epochs : array of floats, size ExCxS, where E is the number of epochs, C is the number of channels, and S is the number of samples within the epoch
        Array containing the EEG data in volts from each of the electrode channels organized by periods of time in which an event (12Hz or 15Hz flashes) occurs.
    fs : array of float, size 1
        The sampling frequency of the data obtained in the 'fs' key of data_dict.

    Returns
    -------
    eeg_epochs_fft : array of complex numbers, size ExCx((fs/2)+1), where E is the number of epochs, C is the number of channels, and fs is the sampling frequency
        The EEG data converted to the frequency space for each epoch and channel.
    fft_frequencies : array of floats, size (fs/2)+1, where fs is the sampling frequency
        Array containing sample frequencies.

    """
  
    # take the Fourier Transform of the epoched EEG data
    eeg_epochs_fft = np.fft.rfft(eeg_epochs)
    
    # find the corresponding frequencies from the epoched EEG data
    fft_frequencies = np.fft.rfftfreq(n=eeg_epochs.shape[-1], d=1/fs) # n is the number of samples in the signal (final dimension) in eeg_epochs), d is the inverse of sampling frequency
  
    return eeg_epochs_fft, fft_frequencies

#%% Get the power spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, class_labels):
    
    # calculate power spectra
    # isolate frequency spectra by event type (12Hz or 15Hz)
    event_high_frequency = eeg_epochs_fft[is_trial_15Hz,:,:]
    event_low_frequency = eeg_epochs_fft[~is_trial_15Hz,:,:]
    
    # calculate power for event type
    event_high_frequency_power = (np.abs(event_high_frequency))**2
    event_low_frequency_power = (np.abs(event_low_frequency))**2 
    
    # calculate mean power for event type
    event_high_frequency_power_mean = event_high_frequency_power.mean(0)
    event_low_frequency_power_mean = event_low_frequency_power.mean(0)
    
    # find maximum power by channel
    event_high_frequency_normalization_factor = event_high_frequency_power_mean.max(1)
    event_low_frequency_normalization_factor = event_low_frequency_power_mean.max(1)
    
    # calculate normalized power for event type
    # preallocate arrays    
    normalized_event_high_frequency_power_mean = np.zeros(event_high_frequency_power_mean.shape)
    normalized_event_low_frequency_power_mean = np.zeros(event_low_frequency_power_mean.shape)

#%% Plot the power spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject, is_plotting=True):
    # normalize to max (all in a channel) - uses the given input if not None
    for channel_index in range(len(channels)):
        
        normalized_event_high_frequency_power_mean[channel_index,:] = event_high_frequency_power_mean[channel_index,:]/event_high_frequency_normalization_factor[channel_index]
        normalized_event_low_frequency_power_mean[channel_index,:] = event_low_frequency_power_mean[channel_index,:]/event_low_frequency_normalization_factor[channel_index]
    
    # calculate spectra for event type
    spectrum_db_15Hz = 10*(np.log10(normalized_event_high_frequency_power_mean))
    spectrum_db_12Hz = 10*(np.log10(normalized_event_low_frequency_power_mean))
    
    # plotting
    if is_plotting == True:
        
        # isolate channel being plotted
        channel_to_plot = [channels.index(channel_name) for channel_name in channels_to_plot]
        
        # set up figure
        figure, channel_plot = plt.subplots(len(channels_to_plot), sharex=True)
        
        for plot_index, channel in enumerate(channel_to_plot): # plot_index to access a subplot
            
            # plot the power spectra by event type
            channel_plot[plot_index].plot(fft_frequencies, spectrum_db_12Hz[channel,:], color='red')
            channel_plot[plot_index].plot(fft_frequencies, spectrum_db_15Hz[channel,:], color='green')
            
            # formatting subplot
            channel_plot[plot_index].set_xlim(0,80)
            channel_plot[plot_index].set_xlabel('frequency (Hz)')
            channel_plot[plot_index].tick_params(labelbottom=True) # shows axis values for each subplot when sharex=True, adapted from Stack Overflow (function and keywords)
            channel_plot[plot_index].set_ylabel('power (dB)')
            channel_plot[plot_index].set_title(f'Channel {channels_to_plot[plot_index]}')
            channel_plot[plot_index].legend(['12Hz','15Hz'], loc='best')
            channel_plot[plot_index].grid()
            
            # plot dotted lines at 12Hz and 15Hz
            channel_plot[plot_index].axvline(12, color='red', linestyle='dotted')
            channel_plot[plot_index].axvline(15, color='green', linestyle='dotted')
        
        # format overall plot
        figure.suptitle(f'SSVEP Subject S{subject} Frequency Content')
        figure.tight_layout()
        
        # save image
        plt.savefig(f'SSVEP_S{subject}_frequency_content.png')
        
    return spectrum_db_15Hz, spectrum_db_12Hz 
