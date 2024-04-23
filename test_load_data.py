#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:48:11 2024

@authors: Arthur Dolimier, Nicholas Kent, Claire Leahy, and Aiden Pricer-Coan
"""

#%% Import packages
from load_data import load_eeg_data

#%% Load the data

# Possible subject labels: 'l1b', 'k6b', or 'k3b'
subject_label = 'k3b'

data_dictionary = load_eeg_data(subject=subject_label)
