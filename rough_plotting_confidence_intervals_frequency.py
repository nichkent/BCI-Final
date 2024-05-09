"""
This script provides functions to ..

@author: Claire Leahy

file: rough_plotting_confidence_intervals_frequecny.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""
# Imports
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_eeg_data
from plot_raw_and_bootstrap_data import plot_raw_data, bootstrap_p_values, extract_epochs, fdr_correction, \
    plot_confidence_intervals_with_significance
from plot_epoch_data import epoch_data
from clean_data import remove_nan_values, separate_test_and_train_data, separate_artifact_trials, separate_by_class, \
    make_finite_filter, filter_data, get_envelope
from frequency_spectrum_data import get_frequency_spectrum, get_power_spectra_epoched
from plot_results import average_around_electrodes_epoched


# plotting function

def plot_power_spectrum(spectra_by_class, fft_frequencies, subject, channels_to_plot, class_to_plot=None):
    # Plot multiple channels for one class
    if class_to_plot != None:

        # Declare figure
        plt.figure()

        # Convert channel list to string for later use
        channels_for_labeling = [str(channel) for channel in channels_to_plot]

        for channel in channels_to_plot:
            # Plot the spectra for the given class of the given channels
            plt.plot(fft_frequencies,
                     spectra_by_class[class_to_plot - 1][channel, :])  # Class labels are 1-indexed, Python is 0-indexed

        # Format figure
        plt.legend(channels_for_labeling)
        plt.title(f'Frequency Spectrum of Class {class_to_plot}')
        plt.grid()
        plt.tight_layout()

    else:

        # Plot the classes together for a given channel
        # Set up figure
        figure, channel_plot = plt.subplots(len(channels_to_plot), sharex=True, figsize=(10, 6))

        for plot_index, channel in enumerate(channels_to_plot):  # plot_index to access a subplot

            for class_spectrum in spectra_by_class:
                # Plot the power spectra by class
                channel_plot[plot_index].plot(fft_frequencies, class_spectrum[channel, :])

            # Formatting subplot
            channel_plot[plot_index].set_xlim(0, 50)
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
        plt.savefig(f'MI_S{subject}_frequency_content.png')


def plot_multiple_power_spectra(spectra_by_class1, spectra_by_class2, fft_frequencies1, fft_frequencies2, subject,
                                channels_to_plot, class_to_plot, activity1='Active', activity2='Rest'):
    figure, channel_plot = plt.subplots(len(channels_to_plot), sharex=True, figsize=(10, 6))

    for plot_index, channel in enumerate(channels_to_plot):  # plot_index to access a subplot

        # Plot the power spectra for the two datasets

        channel_plot[plot_index].plot(fft_frequencies1, spectra_by_class1[class_to_plot - 1][channel, :])
        channel_plot[plot_index].plot(fft_frequencies2, spectra_by_class2[class_to_plot - 1][channel, :])

        # Formatting subplot
        channel_plot[plot_index].set_xlim(0, 50)
        channel_plot[plot_index].set_xlabel('frequency (Hz)')
        channel_plot[plot_index].tick_params(labelbottom=True)
        channel_plot[plot_index].set_ylabel('power (dB)')
        channel_plot[plot_index].set_title(f'Channel {channel}')
        channel_plot[plot_index].grid()

    # Format overall plot
    figure.suptitle(f'MI Subject {subject} Frequency Content')
    figure.legend([activity1, activity2])
    figure.tight_layout()


# loading and plotting
# load data
subject_label = 'l1b'
data_dictionary = load_eeg_data(subject=subject_label)

# unpack data
raw_data = data_dictionary['Signal']  # Raw EEG signal
fs = data_dictionary['Sampling Frequency']  # The sampling frequency
class_labels = data_dictionary['Class Label']  # All the class labels
trigger_times = data_dictionary['Start Times']  # Start time of each trial
is_artifact_trial = data_dictionary['Artifact Trials']  # Truth data of artifact in each trial

# get the epochs
eeg_epochs_rest = epoch_data(fs, trigger_times, raw_data, epoch_start_time=0, epoch_end_time=2)
eeg_epochs_motor_imagery = epoch_data(fs, trigger_times, raw_data, epoch_start_time=5, epoch_end_time=7)

# get the frequency data of the epochs (motor imagery)
eeg_epochs_fft_motor_imagery, fft_frequencies_motor_imagery = get_frequency_spectrum(eeg_epochs_motor_imagery, fs)
spectra_by_class_motor_imagery = get_power_spectra_epoched(eeg_epochs_fft_motor_imagery, fft_frequencies_motor_imagery,
                                                           class_labels)
# plot_power_spectrum(eeg_epochs_motor_imagery, fft_frequencies_motor_imagery, spectra_by_class_motor_imagery,
# subject='l1b', channels_to_plot=[28,34], class_to_plot=1)

# get the frequency data of the epochs (resting period)
eeg_epochs_fft_rest, fft_frequencies_rest = get_frequency_spectrum(eeg_epochs_rest, fs)
spectra_by_class_rest = get_power_spectra_epoched(eeg_epochs_fft_rest, fft_frequencies_rest, class_labels)
# plot_power_spectrum(eeg_epochs_fft_rest, fft_frequencies_rest, spectra_by_class_rest, subject='l1b',
# channels_to_plot=[28,34], class_to_plot=1)

plot_multiple_power_spectra(spectra_by_class_motor_imagery, spectra_by_class_rest, fft_frequencies_motor_imagery,
                            fft_frequencies_rest, subject='l1b', channels_to_plot=[28, 34], class_to_plot=4,
                            activity1='Active', activity2='Rest')
plt.show()

# Bootstrap

"""DURATION OF EPOCHS (ACTIVE AND REST) MUST BE THE SAME SO THAT THE FFT DIMENSIONS ARE THE SAME"""
eeg_epochs_fft_active = eeg_epochs_fft_motor_imagery
spectra_by_class_active = spectra_by_class_motor_imagery
fft_frequencies_active = fft_frequencies_motor_imagery
channels_to_plot = [28, 34]
class_to_plot = 4


def plot_power_spectra_confidence_intervals(eeg_epochs_fft_active, eeg_epochs_fft_rest, spectra_by_class_active,
                                            spectra_by_class_rest, fft_frequencies_active, fft_frequencies_rest,
                                            subject, channels_to_plot, class_to_plot, activity1='Active',
                                            activity2='Rest'):
    figure, channel_plot = plt.subplots(len(channels_to_plot), sharex=True, sharey=True, figsize=(10, 6))

    # Calculate the standard error of the mean for the active epochs, averaged over channels
    active_se_mean = np.mean(np.std(eeg_epochs_fft_active, axis=0) / np.sqrt(eeg_epochs_fft_active.shape[0]), axis=1)

    # Calculate the standard error of the mean for the rest epochs, averaged over channels
    rest_se_mean = np.mean(np.std(eeg_epochs_fft_rest, axis=0) / np.sqrt(eeg_epochs_fft_rest.shape[0]), axis=1)

    for plot_index, channel in enumerate(channels_to_plot):  # plot_index to access a subplot

        # Plot the frequency spectra
        channel_plot[plot_index].plot(fft_frequencies_active, spectra_by_class_active[class_to_plot - 1][channel, :])
        channel_plot[plot_index].plot(fft_frequencies_rest, spectra_by_class_rest[class_to_plot - 1][channel, :])

        # Add confidence intervals for the active frequency spectrum
        channel_plot[plot_index].fill_between(fft_frequencies_active,
                                              spectra_by_class_active[class_to_plot - 1][channel, :] - 2 *
                                              active_se_mean[channel],
                                              spectra_by_class_active[class_to_plot - 1][channel, :] + 2 *
                                              active_se_mean[channel], alpha=0.2, label='Target +/- 95% CI')

        # Add confidence intervals for the rest frequency spectrum
        channel_plot[plot_index].fill_between(fft_frequencies_rest,
                                              spectra_by_class_rest[class_to_plot - 1][channel, :] - 2 * rest_se_mean[
                                                  channel],
                                              spectra_by_class_rest[class_to_plot - 1][channel, :] + 2 * rest_se_mean[
                                                  channel], alpha=0.2, label='Rest +/- 95% CI')

        # Formatting subplot
        channel_plot[plot_index].set_xlim(0, 50)
        channel_plot[plot_index].set_xlabel('frequency (Hz)')
        channel_plot[plot_index].tick_params(labelbottom=True)
        channel_plot[plot_index].set_ylabel('power (dB)')
        channel_plot[plot_index].set_title(f'Channel {channel}')
        channel_plot[plot_index].grid()

    # Format overall plot
    figure.suptitle(f'MI Subject {subject} Frequency Content')
    figure.legend([activity1, activity2])
    figure.tight_layout()
