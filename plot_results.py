import numpy as np
from matplotlib import pyplot as plt


def average_around_electrodes(eeg_data, fs, central_electrodes, surrounding_map):
    """
    Averages the EEG data around specified central electrodes using their surrounding electrodes.

    Parameters:
    - eeg_data (numpy.ndarray): The EEG data array with dimensions [electrodes, time].
    - central_electrodes (list): List of central electrodes to average around.
    - surrounding_map (dict): Dictionary mapping each central electrode to a list of surrounding electrodes.

    Plots:
    - Subplots of averaged EEG signals for each central electrode.
    """
    num_plots = len(central_electrodes)
    fig, axes = plt.subplots(nrows=num_plots, figsize=(10, num_plots * 3))

    time = np.arange(eeg_data.shape[1]) / fs

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot

    for idx, electrode in enumerate(central_electrodes):
        surrounding_electrodes = surrounding_map[electrode]
        # Subtract 1 to convert to zero-indexed Python array indices if necessary
        surrounding_indices = [e - 1 for e in surrounding_electrodes]
        averaged_signal = np.mean(eeg_data[surrounding_indices, :], axis=0)

        ax = axes[idx]
        ax.plot(time, averaged_signal)
        ax.set_title(f"Electrode {electrode} (averaged around {surrounding_electrodes})")
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def average_around_electrodes_epoched(eeg_epochs, fs, central_electrodes, surrounding_map, trials):
    """
    Averages the EEG data around specified central electrodes using their surrounding electrodes,
    for episodic data formatted as num_trials x samples_per_epoch x num_channels.

    Parameters:
    - eeg_epochs (numpy.ndarray): The EEG data array with dimensions [num_trials, samples_per_epoch, num_channels].
    - central_electrodes (list): List of central electrodes to average around.
    - surrounding_map (dict): Dictionary mapping each central electrode to a list of surrounding electrodes.

    Plots:
    - Subplots of averaged EEG signals for each trial for each central electrode.
    """
    num_trials = len(trials)
    # samples_per_epoch = eeg_epochs.shape[1]
    num_plots = len(central_electrodes)

    time = np.arange(0, 10, 1 / fs)

    fig, axes = plt.subplots(nrows=num_plots, ncols=num_trials, figsize=(num_trials * 3, num_plots * 3))
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)  # Adjust the axes array for single subplot cases

    for i, electrode in enumerate(central_electrodes):
        surrounding_electrodes = surrounding_map[electrode]
        surrounding_indices = [e - 1 for e in surrounding_electrodes]  # Adjust if your electrode indices are 1-based

        for j, trial in enumerate(trials):
            averaged_signal = np.mean(eeg_epochs[trial, :, surrounding_indices], axis=0)
            ax = axes[i, j]
            ax.plot(time, averaged_signal)
            ax.set_title(f"Trial {trial + 1}, Electrode {electrode}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()