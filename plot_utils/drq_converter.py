# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: plotting helper code is currently being cleaned up

import numpy as np
import pandas as pd
import os
from numpy import genfromtxt

source_path = 'C:\\z\\drq_author'
for subdir, dirs, files in os.walk(source_path):
    for file in files:
        if '_' in file and '.csv' in file:
            full_path = os.path.join(subdir, file)
            d = pd.read_csv(full_path, index_col=0)
            task = file[4:-4]
            for seed in range(1, 11):
                print(seed)
                d_seed = d[d.seed == seed]
                # now basically save them separately
                folder_name = 'drqv2_author_%s_s%d' % (task, seed)
                folder_path = os.path.join(source_path, folder_name)
                os.mkdir(folder_path)
                save_path = os.path.join(folder_path, 'eval.csv')
                d_seed.to_csv(save_path, index=False)
                print('saved to', save_path)

