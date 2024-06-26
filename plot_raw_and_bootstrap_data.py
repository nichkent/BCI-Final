"""
This script provides functions for plotting and bootstrapping raw eeg data.

@author: nicho

file: plot_raw_and_bootstrap_data.py
BME 6710 - Dr. Jangraw
Project 3: Public Dataset Wrangling
"""
# Imports
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection


def plot_raw_data(raw_data, fs, subject_label, class_label, class_labels):
    """
    Plot EEG data for a specific subject and class label.

    This function filters EEG data to only include trials of a specified class and plots this subset. Each trial
    is plotted as a separate line on the graph to allow visualization of differences across trials within the
    same class. The graph is saved as a PNG file.

    Parameters:
        - raw_data <np.array>[TRIALS, DATA_POINTS]: 2D array where each row represents a trial and columns represent data points within that trial.
        - fs <float>: Sampling frequency of the EEG data, used to convert data points to time in seconds.
        - subject_label <str>: Identifier for the subject, used in the title of the plot and to name the output file.
        - class_label <int>: Numeric label of the class to be plotted. Used to filter the trials in the `raw_data`.
        - class_labels <np.array>[TRIALS]: 1D array with the class labels for each trial in `raw_data`, used to identify trials of the specified class.
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
        plt.plot(time, raw_data[index], label=f'Trial {index + 1}')

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


# Bootsrap the data


def bootstrap_p_values(target_data, non_target_data, iterations=1000, ci=95):
    """
    Calculate the bootstrap p-values by resampling combined target and non-target data.

    This function combines target and non-target data, then resamples this combined dataset to create a distribution
    of mean differences under the null hypothesis. It calculates the observed mean difference, performs bootstrap
    resampling to estimate the distribution of mean differences, and computes a p-value based on this distribution.

    Parameters:
        - target_data <np.array>: Array of data from the target condition.
        - non_target_data <np.array>: Array of data from the non-target condition.
        - iterations <int>: Number of bootstrap iterations to perform (default 1000).
        - ci <int>: Confidence interval percentage to compute (default 95).

    Returns: - observed_diff <float>: The observed difference in means between target and non-target data. - p_value
    <float>: The p-value estimating the probability of observing a difference as extreme as the observed,
    under the null hypothesis. - lower_bound <float>: Lower bound of the confidence interval for the mean difference.
    - upper_bound <float>: Upper bound of the confidence interval for the mean difference.
    """
    # Combine target and non-target data into a single array for resampling
    combined_data = np.concatenate((target_data, non_target_data))

    # Calculate the observed difference in means between the target and non-target data
    observed_diff = np.mean(target_data) - np.mean(non_target_data)

    # Initialize an empty list to store differences from each bootstrap iteration
    bootstrap_diffs = []

    # Perform bootstrap resampling
    for _ in range(iterations):
        # Shuffle the combined data to break any existing order
        np.random.shuffle(combined_data)
        # Split the shuffled data back into 'target' and 'non-target' groups
        boot_target = combined_data[:len(target_data)]
        boot_non_target = combined_data[len(target_data):]
        # Calculate the difference in means for this bootstrap sample
        boot_diff = np.mean(boot_target) - np.mean(boot_non_target)
        # Append the result to the list of bootstrap differences
        bootstrap_diffs.append(boot_diff)

    # Calculate the lower and upper bounds of the specified confidence interval
    lower_bound = np.percentile(bootstrap_diffs, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_diffs, 100 - (100 - ci) / 2)
    # Calculate the p-value as the proportion of bootstrap differences at least as extreme as the observed difference
    p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / iterations

    # Return the observed difference, p-value, and confidence interval bounds
    return observed_diff, p_value, lower_bound, upper_bound


def fdr_correction(p_values, alpha=0.05):
    """
    Apply False Discovery Rate (FDR) correction to p-values.

    Args:
     - p_values <np.array>: Array of p-values to correct
     - alpha <float>: Significance level (default is 0.05)

    Returns:
     - rejected <np.array>: Boolean array indicating which hypotheses are rejected
     - corrected_p_values <np.array>: Array of corrected p-values
    """
    # Perform the FDR correction using the fdrcorrection function from the statsmodels library
    rejected, corrected_p_values = fdrcorrection(p_values, alpha=alpha)

    # Return the results: an array indicating which hypotheses are rejected and the corrected p-values
    return rejected, corrected_p_values


# Function to extract epochs
def extract_epochs(data, start_times, duration):
    """
    Extract epochs from the continuous data starting at specified times with a given duration.

    This function slices the continuous data array to extract segments (epochs) starting from each specified start
    time and continuing for the specified duration.

    Parameters:
        - data <np.array>: The continuous data from which to extract epochs.
        - start_times <np.array>: An array of start times (in samples) for each epoch.
        - duration <int>: The duration (in samples) of each epoch.

    Returns:
        - epochs <np.array>: A 2D array where each row represents an epoch extracted from the continuous data.
    """
    # Create an array of epochs by slicing the continuous data at each start time for the specified duration
    return np.array([data[max(0, start): start + duration] for start in start_times])


def plot_confidence_intervals_with_significance(target_erp, rest_erp, erp_times, target_epochs, rest_epochs,
                                                corrected_p_values, subject_label, class_labels, class_label=None,
                                                channels=None):
    """
    Plot ERPs with confidence intervals and significance markers.

    This function plots event-related potentials (ERPs) for target and rest conditions with confidence intervals. It
    also marks significant differences based on corrected p-values.

    Generates a plot with ERPs, confidence intervals, and significance markers, displayed with appropriate
    labels and legends.

    Parameters:
        - target_erp <np.array>: The average ERP for the target condition.
        - rest_erp <np.array>: The average ERP for the rest condition.
        - erp_times <np.array>: Array of time points corresponding to the ERP data.
        - target_epochs <np.array>: The ERP data for all epochs in the target condition.
        - rest_epochs <np.array>: The ERP data for all epochs in the rest condition.
        - corrected_p_values <np.array>: Array of p-values corrected for multiple comparisons.
        - subject_label <str>: Identifier for the subject being plotted.
        - class_labels <np.array>: Array containing the class labels of the data.
        - class_label <int>: Optional input with which class will be evaluated. The default is None.
        - channels <list>: Optional input for the electrodes the user wishes to evaluate. The default is None.
    """

    # Update to evaluate one class label
    if class_label != None:
        target_class = np.where(class_labels == class_label)[0]  # Take the first index of the tuple
        target_epochs = target_epochs[target_class]  # Update with one class label

    # Update to evaluate specific channels
    if channels != None:
        target_epochs = target_epochs[:, :, channels]  # Take all data across the given channnels

    # Calculate the standard error of the mean for the target ERP across epochs
    target_se_mean = np.std(target_epochs, axis=0) / np.sqrt(target_epochs.shape[0])
    # Calculate the standard error of the mean for the rest ERP across epochs
    rest_se_mean = np.std(rest_epochs, axis=0) / np.sqrt(rest_epochs.shape[0])

    # If there are multiple channels, average the standard errors across channels
    if target_epochs.shape[2] > 1:  # Check if there are multiple channels
        target_se_mean = np.mean(target_se_mean, axis=1)
        rest_se_mean = np.mean(rest_se_mean, axis=1)

    # Set up the plot
    plt.figure(figsize=(10, 5))

    # Plot the target ERP with a label
    plt.plot(erp_times, target_erp, label='Target ERP')

    # Plot the rest ERP with a label
    plt.plot(erp_times, rest_erp, label='Rest ERP')

    # Add confidence intervals for the target ERP
    plt.fill_between(erp_times, target_erp - 2 * target_se_mean, target_erp + 2 * target_se_mean, alpha=0.2,
                     label='Target +/- 95% CI')

    # Add confidence intervals for the rest ERP
    plt.fill_between(erp_times, rest_erp - 2 * rest_se_mean, rest_erp + 2 * rest_se_mean, alpha=0.2,
                     label='Rest +/- 95% CI')

    # Add plot labels and legend
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'Subject {subject_label} ERP Significance')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


def plot_erp_by_class(raw_data, separated_trigger_times, class_labels, subject_label, epoch_duration=1750):
    """
    Plots Event-Related Potentials (ERPs) for different classes and analyzes statistical
    differences between them.

    Parameters:
    - raw_data <np.ndarray>: The raw EEG data array.
    - separated_trigger_times <list of lists>: A list where each sublist contains the trigger times
      for a specific class.
    - class_labels <list of str>: Names of the classes corresponding to each sublist in `separated_trigger_times`.
    - subject_label <str>: Identifier for the subject being analyzed.
    - epoch_duration <int, optional>: Duration of each epoch in milliseconds. Defaults to 1750.
    """

    # Get epochs by class
    class_epochs = []
    for class_start_times in separated_trigger_times:
        class_epochs.append(extract_epochs(raw_data, class_start_times, epoch_duration))

    # Get ERPs by class
    class_erps = []
    for class_epoch in class_epochs:
        class_erps.append(np.mean(class_epoch, axis=(0, 2)))
    erp_times_classes = np.linspace(0, epoch_duration, num=int(epoch_duration))

    # P-values between classes
    p_values_classes = []
    for class_to_compare_index1 in range(4):  # Only use 4 classes, 5th is test data
        for class_to_compare_index2 in range(4):  # Only use 4 classes, 5th is test data
            # Calculate p-value between different classes (only do one time each)
            if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
                p_values_classes.append(
                    bootstrap_p_values(class_epochs[class_to_compare_index1], class_epochs[class_to_compare_index2]))

    # Corrected p-values between classes
    corrected_p_values_classes = []
    for p_value in p_values_classes:
        _, corrected_p_values = fdr_correction(p_value, alpha=0.05)
        corrected_p_values_classes.append(corrected_p_values)

    # Plot the classes
    # NOTE: As function is written, comparison 1 is "target" and comparison 2 is "rest"
    comparison_number = 0
    for class_to_compare_index1 in range(4):  # Only use 4 classes, 5th is test data
        for class_to_compare_index2 in range(4):  # Only use 4 classes, 5th is test data
            if class_to_compare_index1 != class_to_compare_index2 and class_to_compare_index1 < class_to_compare_index2:
                plot_confidence_intervals_with_significance(class_erps[class_to_compare_index1],
                                                            class_erps[class_to_compare_index2], erp_times_classes,
                                                            class_epochs[class_to_compare_index1],
                                                            class_epochs[class_to_compare_index2],
                                                            corrected_p_values_classes[comparison_number],
                                                            subject_label,
                                                            class_labels)

                comparison_number += 1
