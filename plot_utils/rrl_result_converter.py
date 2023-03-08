# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: plotting helper code is currently being cleaned up

# for each result file in each rrl data folder
# load them and convert into a format we can use easily...
import os

import numpy as np
import pandas
from numpy import genfromtxt

base_path = 'C:\\z\\abl'

for subdir, dirs, files in os.walk(base_path):
    if 'log.csv' in files:
        # folder name is here is typically also the variant name
        folder_name = os.path.basename(os.path.normpath(subdir))  # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
        original_log_path = os.path.join(subdir, 'log.csv')

        seed_string_index = folder_name.rfind('s')
        name_no_seed = folder_name[:seed_string_index - 1]
        seed_value = int(folder_name[seed_string_index + 1:])

        original_data = genfromtxt(original_log_path, dtype=float, delimiter=',', names=True)
        iters = original_data['iteration']
        sr = original_data['success_rate']
        # original_data['frames'] = iters * 40000
        new_df = {
            'iter': np.arange(len(iters)),
            'frame': iters * 40000,
            'success_rate': sr/100
        }
        new_df = pandas.DataFrame(new_df)
        save_path = os.path.join(subdir, 'eval.csv')
        new_df.to_csv(save_path, index=False)
        save_path = os.path.join(subdir, 'train.csv')
        new_df.to_csv(save_path, index=False)




