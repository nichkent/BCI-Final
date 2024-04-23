# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:34:26 2024

@author: nicho
"""

import loadmat
data_directory = "data/l1b.mat"

data_file = f"{data_directory}"
data = loadmat.loadmat(data_file)

print(data['HDR']['FileName'])