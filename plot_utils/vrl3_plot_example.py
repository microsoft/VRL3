# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from vrl3_plot_helper import plot_aggregate_0506

##############################################################################
########## change base_dir to where you downloaded the example logs ##########
##############################################################################
base_dir = '/home/watcher/Desktop/projects/vrl3examplelogs'
base_save_dir = '/home/watcher/Desktop/projects/vrl3figures'

FRAMEWORK_NAME = 'VRL3'

NUM_MILLION_ON_X_AXIS = 4
DEFAULT_Y_MIN = -0.02
DEFAULT_X_MIN = -50000
DEFAULT_Y_MAX = 1.02
DEFAULT_X_MAX = 4050000
X_LONG = 12050000
ADROIT_ALL_ENVS =  ["door", "pen", "hammer","relocate"]
adroit_success_rate_default_min_max_dict = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX}
adroit_other_value_default_min_max_dict = {'ymin':None, 'ymax':None, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX}

def decide_placeholder_prefix(envs, folder_name, plot_name):
    if isinstance(envs, list):
        if len(envs) > 1:
            return 'aggregate-' + folder_name, 'aggregate_' + plot_name
        else:
            return folder_name,plot_name +'_'+ envs[0]
    return folder_name, plot_name +'_'+ envs

def plot_paper_main_more_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'adroit_main')
    # below are data folders under the base logs folder, the program will try to find training logs under these folders
    list_of_data_dir = ['rrl', 'vrl3']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'RRL']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['rrl'],
    ]
    colors = ['tab:red', 'tab:grey', 'tab:blue', 'tab:orange', 'tab:pink','tab:brown']
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)

def plot_paper_main_more_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_main_more_aggregate(envs=[env], no_legend=(env!='relocate'))

plot_paper_main_more_per_task()
plot_paper_main_more_aggregate()