"""
This script provides functions to transform eeg data into power and frequency spectra.

@author: Claire Leahy

file: frequency_spectrum_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""

# Import packages

import numpy as np
from matplotlib import pyplot as plt


# Get the frequency spectrum of the data


def get_frequency_spectrum(eeg_epochs, fs):
    # Reshape the epoched data so samples occupy last axis
    reshaped_eeg_epochs = eeg_epochs.transpose(0, 2, 1)  # Shape (epochs, channels, samples)

    # take the Fourier Transform of the epoched EEG data
    eeg_epochs_fft = np.fft.rfft(reshaped_eeg_epochs)

    # find the corresponding frequencies from the epoched EEG data
    fft_frequencies = np.fft.rfftfreq(n=reshaped_eeg_epochs.shape[-1], d=1 / fs)  # n is the number of samples in the
    # signal (final dimension) in eeg_epochs), d is the inverse of sampling frequency

    return eeg_epochs_fft, fft_frequencies


# Get the power spectra


def get_power_spectra_epoched(eeg_epochs_fft, fft_frequencies, class_labels):
    # print(eeg_epochs_fft.shape)

    # Sort class labels (take first index of tuple)
    class1 = np.where(class_labels == 1)[0]
    class2 = np.where(class_labels == 2)[0]
    class3 = np.where(class_labels == 3)[0]
    class4 = np.where(class_labels == 4)[0]
    test_class = np.where(np.isnan(class_labels))[0]  # NaN assigned to test data

    # Calculate power spectra
    # Take FFT by class label
    class1_frequency = eeg_epochs_fft[class1, :, :]
    class2_frequency = eeg_epochs_fft[class2, :, :]
    class3_frequency = eeg_epochs_fft[class3, :, :]
    class4_frequency = eeg_epochs_fft[class4, :, :]
    test_class_frequency = eeg_epochs_fft[test_class, :, :]

    # Calculate power for class
    class1_power = (np.abs(class1_frequency)) ** 2
    class2_power = (np.abs(class2_frequency)) ** 2
    class3_power = (np.abs(class3_frequency)) ** 2
    class4_power = (np.abs(class4_frequency)) ** 2
    test_class_power = (np.abs(test_class_frequency)) ** 2

    # Calculate mean power for class
    class1_power_mean = class1_power.mean(0)
    class2_power_mean = class2_power.mean(0)
    class3_power_mean = class3_power.mean(0)
    class4_power_mean = class4_power.mean(0)
    test_class_power_mean = test_class_power.mean(0)

    # Find maximum power by channel
    class1_normalization_factor = class1_power_mean.max(1)
    class2_normalization_factor = class2_power_mean.max(1)
    class3_normalization_factor = class3_power_mean.max(1)
    class4_normalization_factor = class4_power_mean.max(1)
    test_class_normalization_factor = test_class_power_mean.max(1)

    # Calculate normalized power for event type
    # Preallocate arrays    
    normalized_class1_power_mean = np.zeros(class1_power_mean.shape)
    normalized_class2_power_mean = np.zeros(class2_power_mean.shape)
    normalized_class3_power_mean = np.zeros(class3_power_mean.shape)
    normalized_class4_power_mean = np.zeros(class4_power_mean.shape)
    normalized_test_class_power_mean = np.zeros(test_class_power_mean.shape)

    # Normalize to max (all in a channel) - uses the given input if not None
    channel_count = eeg_epochs_fft.shape[1]  # Second index is number of channels
    for channel_index in range(channel_count):
        normalized_class1_power_mean[channel_index, :] = class1_power_mean[channel_index, :] / \
                                                         class1_normalization_factor[channel_index]
        normalized_class2_power_mean[channel_index, :] = class2_power_mean[channel_index, :] / \
                                                         class2_normalization_factor[channel_index]
        normalized_class3_power_mean[channel_index, :] = class3_power_mean[channel_index, :] / \
                                                         class3_normalization_factor[channel_index]
        normalized_class4_power_mean[channel_index, :] = class4_power_mean[channel_index, :] / \
                                                         class4_normalization_factor[channel_index]
        normalized_test_class_power_mean[channel_index, :] = test_class_power_mean[channel_index, :] / \
                                                             test_class_normalization_factor[channel_index]

    # Calculate spectra for event type
    spectrum_db_class1 = 10 * (np.log10(normalized_class1_power_mean))
    spectrum_db_class2 = 10 * (np.log10(normalized_class2_power_mean))
    spectrum_db_class3 = 10 * (np.log10(normalized_class3_power_mean))
    spectrum_db_class4 = 10 * (np.log10(normalized_class4_power_mean))
    spectrum_db_test_class = 10 * (np.log10(normalized_test_class_power_mean))

    # spectrum_db_class1 = 10*(np.log10(class1_power_mean))
    # spectrum_db_class2 = 10*(np.log10(class2_power_mean))
    # spectrum_db_class3 = 10*(np.log10(class3_power_mean))
    # spectrum_db_class4 = 10*(np.log10(class4_power_mean))
    # spectrum_db_test_class = 10*(np.log10(test_class_power_mean))

    # Create a list to return spectra together
    spectra_by_class = [spectrum_db_class1, spectrum_db_class2, spectrum_db_class3, spectrum_db_class4,
                        spectrum_db_test_class]

    return spectra_by_class


# Plot the power spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, spectra_by_class, channels, subject):
    # Set up figure
    figure, channel_plot = plt.subplots(len(channels), sharex='True', figsize=(10, 6))

    for plot_index, channel in enumerate(channels):  # plot_index to access a subplot

        for class_spectrum in spectra_by_class:
            # Plot the power spectra by class
            channel_plot[plot_index].plot(fft_frequencies, class_spectrum[channel, :])

        # Formatting subplot
        channel_plot[plot_index].set_xlim(0, 35)
        channel_plot[plot_index].set_xlabel('frequency (Hz)')
        channel_plot[plot_index].tick_params(labelbottom=True)
        channel_plot[plot_index].set_ylabel('power (dB)')
        channel_plot[plot_index].set_title(f'Channel {channel}')
        channel_plot[plot_index].grid()

    # Format overall plot
    figure.suptitle(f'MI Subject S{subject} Frequency Content')
    figure.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Test'])
    figure.tight_layout()

    # save image
    plt.savefig(f'MI_{subject}_frequency_content.png')


def get_power_spectra_single(epoch_fft, sfreq):
    # Calculate power spectrum
    power = np.abs(epoch_fft) ** 2

    # Find maximum power by channel
    normalization_factor = power.max(0)

    normalized_power_mean = np.zeros(power.shape)

    channel_count = epoch_fft.shape[0]
    for channel_index in range(channel_count):
        normalized_power_mean[channel_index, :] = power[channel_index, :] / normalization_factor[channel_index]

    spectrum_db = 10 * (np.log10(normalized_power_mean))

    return spectrum_db


def plot_power_spectrum_single(fft_frequencies, spectrum, class_label, channels, subject, epoch_index):
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(channels):
        plt.plot(fft_frequencies, spectrum[channel, :], label=f'Channel {channel}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xlim([0, 35])
    plt.title(f'Power Spectrum of Epoch {epoch_index} {class_label} for {subject}')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_frequency_spectrum_single(epoch, fs):
    # Reshape the epoched data so samples occupy last axis
    reshaped_eeg_epochs = epoch.transpose(1, 0)  # Shape (channels, samples)

    # take the Fourier Transform of the epoched EEG data
    eeg_epoch_fft = np.fft.rfft(reshaped_eeg_epochs)

    # find the corresponding frequencies from the epoched EEG data
    fft_frequencies = np.fft.rfftfreq(n=reshaped_eeg_epochs.shape[-1],
                                      d=1 / fs)  # n is the number of samples in the signal (final dimension) in
    # eeg_epochs), d is the inverse of sampling frequency

    return eeg_epoch_fft, fft_frequencies
