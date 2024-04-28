# README for 4-class data for BCI competition 2005; compiled by A. Schloegl

Writen 4/23/24 by Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan

# Overview:
This dataset is part of the BCI Competition 2005 and involves EEG data for analyzing motor imagery. It includes EEG recordings from three participants who imagined movements of four body parts: the tongue, left hand, right hand, and feet. Each participant completed at least six runs containing 40 trials each. The data aims to facilitate the development and benchmarking of algorithms that can predict a user's intent based on EEG signals.

# Experiment Protocol:
Each trial begins with two seconds of silence followed by an acoustic stimulus at t = 2 seconds signaling the start. A fixation cross "+" appears at t = 3 seconds alongside an arrow pointing in the direction corresponding to one of the four imagined movements. Participants were asked to imagine the movement (left hand, right hand, tongue, or foot) until the cross disappeared at t = 7 seconds. Each movement was cued 10 times per run in a randomized order.

# Data Format and Access:
The EEG data was downloaded in MATLAB format, making it accessible via the MATLAB environment or tools that support MATLAB file formats. To download the data, please visit the following link and agree to the dataset terms: https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit "4-class data for BCI competition 2005; compiled by A. Schloegl".

## Loading the Data:
There are three subject files total. The data comes in a zip file from the provided link. After unzipping the file, the data can be loaded using loadmat.py provided by Dr. Jangraw. Below is the code used to load the data:
```python
import loadmat

data_file = "path_to_data.mat"

data = loadmat.loadmat(data_file)
```

The suggested method for formatting the data is as follows. The format that the data comes in is a bit non-intuitive. The data comes loaded into a dictionary. Within this dictionary is another dictionary titled HDR, from the HDR dictionary you can extract the fields as such:

```python
signal = data['s'] # EEG signal data array with potential NaN values indicating breaks or data saturation.

fs = data['HDR']['SampleRate'] # Sampling rate
class_label = data['HDR']['Classlabel'] # Labels of each class (left hand=1, right hand=2, foot=3, tongue=4) and NAN (trials of the test set)
trigger_time = data['HDR']['TRIG'] # Start time of each trial
artifact_selection = data['HDR']['ArtifactSelection'] # Indicates trials with artifacts which were visually identified
```

# Data Structure:
The data appears to have many fields that are not relevant to the EGG data. As such the following fields are the only ones relevant to our research:
- 's': 2D array (time points x channels), EEG signal data array with potential NaN values indicating break or data saturation.
- 'SampleRate': 1D array of sample rates listed for each trial. Always 250 Hz/second.
- 'Classlabel': 1D array, numeric labels (1-4) indicating the movement class for each trial; NaN values denote test set trials.
- 'TRIG': 1D array, start time for each trial.
- 'ArtifactSelection': 1D array, indicates trials with artifacts which were visually identified.

# Evaluation:
Participants are expected to output continuous classification results for all four classes throughout each trial. A confusion matrix in the original study was built for each time-point between 0.0 s and 7.0 s, using which the accuracy and kappa coefficient are derived.

# Notes: 
- While the hands were deemed as separate classes in the data the feet were recorded together.
- Only the SampleRate, Classlabel, TRIG, and ArtifactSelection fields are pertinent for typical usage focused on classifying the motor imagery data.