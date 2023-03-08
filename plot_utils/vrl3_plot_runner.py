# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: plotting helper code is currently being cleaned up

import copy
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns

from vrl3_plot_helper import plot, hyper_search, combine_data_in_seeds, plot_aggregate, plot_aggregate_0506, hyper_search_adroit, plot_hyper_sensitivity, hyper_search_dmc
from plot_config import DEFAULT_BASE_SAVE_PATH, DEFAULT_AMLT_PATH, DEFAULT_BASE_DATA_PATH

base_dir = DEFAULT_BASE_DATA_PATH
base_save_dir = DEFAULT_BASE_SAVE_PATH

FRAMEWORK_NAME = 'VRL3'
DMC_HARD_ENVS = ["humanoid_stand", "humanoid_walk", "humanoid_run"]

def expand_with_env_name(variant2dict, env):
    new_dict = {}
    for key, value in variant2dict.items():
        new_dict[key + '_' + env] = value
    return new_dict

def plot_compare_out_compress():
    list_of_data_dir = ['base_easy', 'fixed1', ]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128',

        'drq_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nstep3_pt5_0.001_ftrue_numc32',
        'drq_nstep3_pt5_0.001_ftrue_numc64',
        # 'drq_nstep3_pt5_0.001_ftrue_numc96',
        'drq_nstep3_pt5_0.001_ftrue_numc128',
        # 'drq_nstep3_pt5_0.001_ftrue_numc192',

    ]

    variant2dashes = {
        'drq_nstep3_pt5_0.001_ftrue_numc2':True,
        'drq_nstep3_pt5_0.001_ftrue_numc32': True,
        'drq_nstep3_pt5_0.001_ftrue_numc64': True,
        'drq_nstep3_pt5_0.001_ftrue_numc128': True,

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128': False,
    }

    variant2colors = {
        'drq_nstep3_pt5_0.001_ftrue_numc2': 'tab:gray',
        'drq_nstep3_pt5_0.001_ftrue_numc32': 'tab:blue',
        'drq_nstep3_pt5_0.001_ftrue_numc64': 'tab:orange',
        'drq_nstep3_pt5_0.001_ftrue_numc128': 'tab:red',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'tab:red',
    }

    variant2labels = {
        'drq_nstep3_pt5_0.001_ftrue_numc2': 'compress_out_numc2',
        'drq_nstep3_pt5_0.001_ftrue_numc32': 'compress_out_numc32',
        'drq_nstep3_pt5_0.001_ftrue_numc64': 'compress_out_numc64',
        'drq_nstep3_pt5_0.001_ftrue_numc128': 'compress_out_numc128',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'conv_out_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'conv_out_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'conv_out_numc64',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'conv_out_numc128',
    }

    for env in [
        "cartpole_swingup", "cheetah_run", "walker_walk",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=True,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             variant2labels=expand_with_env_name(variant2labels, env)
             )

def plot_compare_out_compress2():
    # TODO I haven't saved results here yet
    #  compare freeze versus no freeze, curl conv encoder, change number of channel
    #  it seems on walker, no-freeze works a bit better

    list_of_data_dir = [#'base_easy',
                        'fixed1','drq-base-pretanh' ]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',
        'drq_nstep3_pt5_0.001',

        'drq_nec_nstep3_pt5_0.001_ffalse_numc2',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc32',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc64',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc128',

        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc16',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc128'
    ]

    variant2dashes = {
        'drq_nstep3_pt5_0.001':False,
        'drq_nec_nstep3_pt5_0.001_ffalse_numc2':True,
        'drq_nec_nstep3_pt5_0.001_ffalse_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ffalse_numc64':True,
        'drq_nec_nstep3_pt5_0.001_ffalse_numc128':True,

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128': False,
    }

    variant2colors = {
        'drq_nstep3_pt5_0.001':'tab:purple',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc32': 'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc64': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ffalse_numc128': 'tab:red',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'tab:red',
    }

    variant2labels = {
        'drq_nstep3_pt5_0.001_ftrue_numc2': 'compress_out_numc2',
        'drq_nstep3_pt5_0.001_ftrue_numc32': 'compress_out_numc32',
        'drq_nstep3_pt5_0.001_ftrue_numc64': 'compress_out_numc64',
        'drq_nstep3_pt5_0.001_ftrue_numc128': 'compress_out_numc128',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'conv_out_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'conv_out_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'conv_out_numc64',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'conv_out_numc128',
    }

    for env in [
        "walker_walk",
        # "cheetah_run",
        # "cartpole_swingup",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env)
             )

    for env in [
        "walker_walk",
        # "cheetah_run",
        # "cartpole_swingup",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_y_value='pretanh',
             plot_extra_y=False,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env)
             )

def plot_resnet_new_cap():
    list_of_data_dir = ['base_easy', 'resnet-new', ]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',

        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap3_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap5_numc32',

        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc8',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc16',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc96',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc8',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc16',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc96',
    ]

    variant2dashes = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32':False,

        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap3_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap5_numc32':True,
    }

    variant2colors = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32': 'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32': 'tab:red',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32': 'tab:brown',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32': 'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap3_numc32': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32': 'tab:red',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap5_numc32': 'tab:brown',
    }

    variant2labels = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32': 'resnet6_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32': 'resnet8_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32': 'resnet10_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32': 'resnet18_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32': 'resnet34_additional',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32': 'resnet6',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32': 'resnet8',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap3_numc32': 'resnet10',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32': 'resnet18',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap5_numc32': 'resnet34',
    }

    for env in [
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk", "hopper_hop", "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )

def plot_resnet_new_numc():
    list_of_data_dir = ['base_easy', 'resnet-new', ]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',

        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc8',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc96',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc8',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc96',
    ]



    variant2dashes = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc2':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc8':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc16':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32':False,
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc64':False,

        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc2':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc8':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc16':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32':True,
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc64':True,
    }

    variant2colors = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc8': 'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc16': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32': 'tab:red',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc64': 'tab:brown',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc8': 'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc16': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32': 'tab:red',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc64': 'tab:brown',
    }

    variant2labels = {
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc2': 'resnet_numc2_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc8': 'resnet_numc8_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc16': 'resnet_numc16_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32': 'resnet_numc32_additional',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc64': 'resnet_numc64_additional',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc2': 'resnet_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc8': 'resnet_numc8',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc16': 'resnet_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32': 'resnet_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc64': 'resnet_numc64',
    }

    for env in [
        "cartpole_swingup", "cheetah_run", "walker_walk",
        # "cartpole_swingup", "cheetah_run",
        "quadruped_walk", "hopper_hop", "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )


def plot_resnet_compare_to_old_structure():
    list_of_data_dir = ['base_easy', 'resnet-new', 'fixed1']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',

        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc128',

        # 'drq_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nstep3_pt5_0.001_ftrue_numc32',
        # 'drq_nstep3_pt5_0.001_ftrue_numc64',
        # 'drq_nstep3_pt5_0.001_ftrue_numc96',
        # 'drq_nstep3_pt5_0.001_ftrue_numc128',
        # 'drq_nstep3_pt5_0.001_ftrue_numc192',


        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32',

        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap2_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap3_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_cap5_numc32',
    ]

    variant2labels ={
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':'curl_no_out_compress',
        'drq_nstep3_pt5_0.001_ftrue_numc32':'curl_compress',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32':'resnet_additional_cap1',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32':'resnet_additional_cap4',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32':'resnet_cap1',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32':'resnet_cap4',
    }

    for env in [
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "cartpole_swingup", "cheetah_run", "walker_walk",
        # "quadruped_walk", "hopper_hop", "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )


def plot_resnet_end_end():
    list_of_data_dir = ['resnet-end-end', 'resnet-new']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap0_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32',
        'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32',
        'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32',
    ]

    variant2labels ={
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':'curl_no_out_compress',
        'drq_nstep3_pt5_0.001_ftrue_numc32':'curl_compress',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32':'resnet_additional_cap1',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32':'resnet_additional_cap4',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap1_numc32':'resnet_cap1',
        'drq_nec_nstep3_pt5_0.001_ftrue_cap4_numc32':'resnet_cap4',
    }

    for env in [
        "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk", "hopper_hop", "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )


def plot_fiencoder():
    list_of_data_dir = ['fiencoder', 'base_easy', 'resnet-new', 'fixed1']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_nstep3_pt5_0.001',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap0_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap1_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap3_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap4_numc32',
        # 'drq_nec_qgrad_nstep3_pt5_0.001_ftrue_aotrue_cap5_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_aotrue_cap2_numc32',

        'drq_fi_nstep3_pt5_0.001_ftrue_aofalse_cap1_numc32',
        'drq_fi_nstep3_pt5_0.001_ftrue_aofalse_cap3_numc32',
        'drq_fi_nstep3_pt5_0.001_ftrue_aofalse_cap5_numc32'
    ]

    for env in [
        # "humanoid_walk",
        "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk", "hopper_hop", "finger_turn_hard",

    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )

def plot_resbase():
    list_of_data_dir = ['resbase']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        'drq_n3_p0.001_encul_atc_cap1_nc32_l512',
        'drq_n3_p0.001_encul_atc_cap3_nc32_l512',
        'drq_n3_p0.001_encul_atc_cap4_nc32_l512',

        'drq_n3_p0.001_encboth_atc_cap1_nc32_l512',
        'drq_n3_p0.001_encboth_atc_cap3_nc32_l512',
        'drq_n3_p0.001_encboth_atc_cap4_nc32_l512',
        # 'drq_n3_p0.001_encrl_atc_cap4_nc32_l512',
        # 'drq_n3_p0.001_encul_atc_cap4_nc32_l512',
        # 'drq_n3_p0.001_encboth_atc_cap4_nc32_l512',
    ]

    variant2colors = {
        'drq_n3_p0.001_encul_atc_cap1_nc32_l512':'tab:olive',
        'drq_n3_p0.001_encul_atc_cap3_nc32_l512':'tab:orange',
        'drq_n3_p0.001_encul_atc_cap4_nc32_l512':'tab:red',

        'drq_n3_p0.001_encboth_atc_cap1_nc32_l512':'tab:gray',
        'drq_n3_p0.001_encboth_atc_cap3_nc32_l512':'tab:cyan',
        'drq_n3_p0.001_encboth_atc_cap4_nc32_l512':'tab:blue',

    }

    for env in [
        # "humanoid_walk",
        "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk", "hopper_hop", "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )
def plot_atc():
    list_of_data_dir = ['atc', 'resbase']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]

    # 50-50 3 modes compare, they seem to be similar?
    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    # ]

    # for each mode, compare different sizes
    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc512_rc50',
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc50_rc512',
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc512_rc512',
    # ]

    # to_plot = [
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc50',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc512',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc512',
    # ]
    #
    to_plot = [
        'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
        'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc50',
        'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc512',
        'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc512',
    ]

    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc512_rc512',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc512',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc512',
    # ]
    #
    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc512_rc50',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc50',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc50',
    # ]
    #
    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc50_rc512',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc512',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc512',
    # ]
    #
    # to_plot = [
    #     'drq_n3_p0.001_encrl_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    # ]

    to_plot = [
    'drq_n3_p0.001_encul_atc_cap1_nc32_l512',
    'drq_n3_p0.001_encul_atc_cap3_nc32_l512',
    'drq_n3_p0.001_encul_atc_cap4_nc32_l512',

    'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc50',
    'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc50',
    'drq_n3_p0.001_encul_atc_cap0_nc32_uc50_rc512',
    'drq_n3_p0.001_encul_atc_cap0_nc32_uc512_rc512',
    ]

    to_plot = [
    'drq_n3_p0.001_encboth_atc_cap1_nc32_l512',
    'drq_n3_p0.001_encboth_atc_cap3_nc32_l512',
    'drq_n3_p0.001_encboth_atc_cap4_nc32_l512',

    'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc50',
    'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc512',
    'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc512',
    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "hopper_hop",
        "finger_turn_hard",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )

def plot_atc_dense():
    list_of_data_dir = ['atc', 'resbase', 'drq-base-pretanh']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]

    to_plot = [ # rl versus ul
        'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc50',
    ]

    to_plot = [ # rl versus both
        'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap3_nc32_uc50',
    ]

    to_plot = [
        'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap1_nc32_uc50',
        # 'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc50',
        # 'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc50',
        # 'drq_n3_p0.001_encboth_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap3_nc32_uc50',
    ]

    to_plot = [
        'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap1_nc32_uc50',
        # 'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc50',
        # 'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc50',
        # 'drq_n3_p0.001_encboth_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encboth_dense_atc_cap3_nc32_uc50',
    ]

    to_plot = [ # ul versus both
        'drq_nstep3_pt5_0.001',
        'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc50',
        'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc50',
    ]


    # to_plot = [
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc50',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc512',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc512',
    # #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc512',
    #     'drq_n3_p0.001_encul_dense_atc_cap1_nc32_uc512',
    #     'drq_n3_p0.001_encboth_dense_atc_cap1_nc32_uc512',
    #     'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc512',
    #     'drq_n3_p0.001_encul_dense_atc_cap2_nc32_uc512',
    #     'drq_n3_p0.001_encboth_dense_atc_cap2_nc32_uc512',
    #     'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc512',
    #     'drq_n3_p0.001_encul_dense_atc_cap3_nc32_uc512',
    #     'drq_n3_p0.001_encboth_dense_atc_cap3_nc32_uc512',
    # ]
    #
    # to_plot = [
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc50',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc512',
    # # 'drq_n3_p0.001_encboth_atc_cap0_nc32_uc512_rc512',
    #     'drq_n3_p0.001_encboth_atc_cap0_nc32_uc50_rc50',
    #     'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc50',
    #     'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc50',
    #     'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc50',
    #     'drq_n3_p0.001_encrl_dense_atc_cap1_nc32_uc512',
    #     'drq_n3_p0.001_encrl_dense_atc_cap2_nc32_uc512',
    #     'drq_n3_p0.001_encrl_dense_atc_cap3_nc32_uc512',
    # ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10
             )


def plot_distill_change_weight_and_distill_index():
    list_of_data_dir = ['distill', 'base_easy']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'distill')
    save_prefix = 'distill'

    to_plot = [ # rl versus ul
        # 'drq_nstep3_pt5_0.001',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds1_0.01_res_atc_cap1_nc32',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds1_0.1_res_atc_cap1_nc32',
        'drq_distill_n3_p0.001_encboth_pre100000_ds1_1_res_atc_cap1_nc32',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds3_0.01_res_atc_cap1_nc32',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds3_0.1_res_atc_cap1_nc32',
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap1_nc32',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds5_0.01_res_atc_cap1_nc32',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds5_0.1_res_atc_cap1_nc32',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap1_nc32',
        #
        'drq_distill_n3_p0.001_encboth_pre100000_ds1_0_res_atc_cap1_nc32',

    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        save_name = '%s_%s_%s' % (save_prefix, env, 'performance')
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10,
             save_folder=save_folder,
             save_name=save_name
             )

def plot_distill_low_cap():
    list_of_data_dir = ['distill', 'base_easy']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'distill')
    save_prefix = 'distill_change_numc'

    to_plot = [ # rl versus ul
        # 'drq_nstep3_pt5_0.001',
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap1_nc4',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap1_nc8',
        # 'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap1_nc12',
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap1_nc16',

        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap1_nc4',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap1_nc16',

        'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc4',
        # 'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc8',
        # 'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc12',
        'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc16',

    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        save_name = '%s_%s_%s' % (save_prefix, env, 'performance')
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10,
             save_folder=save_folder,
             save_name=save_name
             )

def plot_distill_change_ul_cap():
    list_of_data_dir = ['distill', 'base_easy']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'distill')
    save_prefix = 'distill_ul_high_cap'

    to_plot = [ # rl versus ul
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap2_nc4',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap3_nc4',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap4_nc4',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap2_nc8',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap3_nc8',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap4_nc8',
    ]

    to_plot = [ # rl versus ul
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap2_nc16',
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap3_nc16',
        'drq_distill_n3_p0.001_encboth_pre100000_ds3_1_res_atc_cap4_nc16',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap2_nc16',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap3_nc16',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap4_nc16',

        # 'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc4',
        # 'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc8',
        # 'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc12',
        'drq_distill_n3_p0.001_encboth_pre0_ds1_0_res_atc_cap1_nc16',
    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        save_name = '%s_%s_%s' % (save_prefix, env, 'performance')
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10,
             save_folder=save_folder,
             save_name=save_name
             )

def plot_compare_distill_to_baseline():
    list_of_data_dir = ['distill', 'drq-base-pretanh']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'distill')
    save_prefix = 'distill_compare_baseline'


    to_plot = [ # rl versus ul
        'drq_nstep3_pt5_0.001',
        'drq_distill_n3_p0.001_encboth_pre100000_ds1_0_res_atc_cap1_nc32',
        'drq_distill_n3_p0.001_encboth_pre100000_ds5_1_res_atc_cap1_nc32',
    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        save_name = '%s_%s_%s' % (save_prefix, env, 'performance')
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10,
             save_folder=save_folder,
             save_name=save_name
             )


def plot_compare_n_compress():
    list_of_data_dir = ['drq-base-pretanh', 'n_compress']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'n_compress')
    save_prefix = 'change_n_compress'

    to_plot = [
        'drq_nstep3_pt5_0.001',
        'drq_single_n3_p0.001_encul_pre100000_comp1_ds5_1_res_atc_cap2_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp2_ds5_1_res_atc_cap2_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp3_ds5_1_res_atc_cap2_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp4_ds5_1_res_atc_cap2_nc32',
    ]

    to_plot = [
        'drq_nstep3_pt5_0.001',
        'drq_single_n3_p0.001_encul_pre100000_comp1_ds5_1_res_atc_cap4_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp2_ds5_1_res_atc_cap4_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp3_ds5_1_res_atc_cap4_nc32',
        'drq_single_n3_p0.001_encul_pre100000_comp4_ds5_1_res_atc_cap4_nc32',
    ]

    for env in [
        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        "quadruped_walk",
        "walker_walk",
        "finger_turn_hard"
    ]:
        print(env)
        save_name = '%s_%s_%s' % (save_prefix, env, 'performance')
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=False,
             # variant2colors=expand_with_env_name(variant2colors, env),
             # variant2dashes=expand_with_env_name(variant2dashes, env),
             # variant2labels=expand_with_env_name(variant2labels, env),
             smooth=10,
             save_folder=save_folder,
             save_name=save_name
             )

def plot_double():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')
    save_prefix = 'double-compare-curl-freeze'

    to_plot = [
        'drq_nstep3_pt5_0.001',
        'drq_double_n3_p0.001_encrl_pre0_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encboth_pre0_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc128',
    ]

    to_plot = [ # with curl freeze
        'drq_nstep3_pt5_0.001',
        'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        'drq_double_n3_p0.001_encrl_pre0_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encboth_pre0_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc32',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc64',
        # 'drq_nec_nstep3_pt5_0.001_ffalse_numc128',
    ]

    # TODO we can make plotting faster...
    #  basically first load all relevant data, then do plotting... or keep a global dictionary, if we load some data, we put into
    #  that dict, so that we don't load again (lazy loading + store in memory)
    for env in [
        # "walker_walk",

        "quadruped_walk", "finger_turn_hard", "hopper_hop", "walker_walk",
        # "cartpole_swingup_sparse", "humanoid_walk",

        # "humanoid_walk",
        # "cartpole_swingup", "cheetah_run", "walker_walk",
        # "quadruped_walk",
        # "walker_walk",
        # "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )


def plot_intermediate_standard_conv():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'intermediate']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')
    save_prefix = 'intermediate-standard-atc'

    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_0_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_2_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_3_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_4_nc32',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
        "quadruped_walk", "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

    save_prefix = 'intermediate-standard-curl'
    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_2_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_3_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_4_nc32',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
        "quadruped_walk", "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_intermediate_standard_conv_freeze():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'intermediate']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')
    save_prefix = 'intermediate-standardfreeze-atc'

    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap0_0_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap0_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap0_2_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap0_3_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap0_4_nc32',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
         "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

    save_prefix = 'intermediate-standardfreeze-curl'
    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_2_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_3_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_4_nc32',
        # 'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
        "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )


def plot_intermediate_resnet():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'intermediate']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')

    save_prefix = 'intermediate-resnet-atc'
    to_plot = [
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_2_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_3_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_4_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_5_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
        "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

    save_prefix = 'intermediate-resnet-curl'
    to_plot = [
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_1_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_2_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_3_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_4_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_5_nc32',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_6_nc32',
    ]

    for env in [
        "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )


def plot_intermediate_resnet_freeze():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'intermediate']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')

    save_prefix = 'intermediate-resnetfreeze-atc'
    to_plot = [
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_2_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_3_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_4_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_5_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_atc_cap1_6_nc32',
    ]

    for env in [
        "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

    save_prefix = 'intermediate-resnetfreeze-curl'
    to_plot = [
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap0_0_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_1_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_2_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_3_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_4_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_5_nc32',
        'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_6_nc32',
    ]

    for env in [
        "walker_walk", "finger_turn_hard"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_double_res_additional():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'double-add']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')

    save_prefix = 'doubleresadd'

    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        'drq_double_n3_p0.001_encboth_pre0_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre0_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue',
    ]

    for env in [
        # "finger_turn_hard",
        # "quadruped_walk",
        # "cheetah_run",
        "quadruped_walk", "walker_walk", "finger_turn_hard", "hopper_hop", "cartpole_swingup_sparse", "cheetah_run"
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )


def plot_double_res_additional_with_baseline():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'double-add']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'double')

    save_prefix = 'doubleresadd-with-base'

    to_plot = [  # with curl freeze
        'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        'drq_double_n3_p0.001_encboth_pre0_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre0_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue',
    ]

    for env in [
        # "finger_turn_hard",
        # "quadruped_walk",
        # "cheetah_run",
        "quadruped_walk", "walker_walk", "finger_turn_hard", "hopper_hop", "cheetah_run", "cartpole_swingup_sparse",
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )



def plot_single_res_policy_loss_gradient():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'double-add', 'test-pg']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pg')
    save_prefix = 'pg'

    to_plot = [ # with curl freeze
        'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addfalse_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addfalse_pgfalse',

        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_nc32_addtrue_pgtrue',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_nc32_addtrue_pgfalse',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_nc32_addfalse_pgtrue',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap1_nc32_addfalse_pgfalse',
    ]

    for env in [
        "finger_turn_hard",
        # "quadruped_walk",
        # "cheetah_run",
    ]:
        print(env)
        for y in ['performance',
                  'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_double_standard_policy_loss_gradient():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'double-add', 'test-pg']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pg')
    save_prefix = 'pg-double-standard'

    to_plot = [ # with curl freeze
        'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addfalse_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addfalse_pgfalse',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap0_nc32_addtrue_pgtrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap0_nc32_addfalse_pgtrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap0_nc32_addtrue_pgfalse',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap0_nc32_addfalse_pgfalse',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap0_nc32_addtrue_pgtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap0_nc32_addfalse_pgtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap0_nc32_addtrue_pgfalse',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap0_nc32_addfalse_pgfalse',
    ]

    for env in [
        "finger_turn_hard",
        # "quadruped_walk",
        # "cheetah_run",
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_naivelarge():
    list_of_data_dir = ['drq-base-pretanh', 'double', 'fixed1', 'curl-freeze', 'double-add', 'naivelarge']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'naivelarge')
    save_prefix = 'naivelarge'

    to_plot = [ # with curl freeze
        # 'drq_nstep3_pt5_0.001',
        # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre0_res_curl_cap1_nc32_addfalse_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addtrue_pgfalse',
        # 'drq_single_n3_p0.001_encnone_pre100000_res_curl_cap1_nc32_addfalse_pgfalse',
        'drq_single_n3_p0.001_encrl_pre0_res_curl_cap3_nc32_addfalse_pgfalse',
        'drq_single_n3_p0.001_encrl_pre0_res_curl_cap4_nc32_addfalse_pgfalse',
        'drq_single_n3_p0.001_encrl_pre0_res_curl_cap5_nc32_addfalse_pgfalse',
    ]

    for env in [
        "quadruped_walk", "walker_walk", "finger_turn_hard", "hopper_hop"
    ]:
        print(env)
        for y in ['performance',
                  'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_medium12_3seed_doubleres_baseline_compare():
    list_of_data_dir = ['newbaseline3s', 'newdouble3s']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'compare3seed')
    save_prefix = 'compare3seed_doubleres_baseline'

    to_plot = [
        'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap4_nc32_addtrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue_qinultrue',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue_qinultrue',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_nc64_addfalse'
    ]
    variant2labels ={
        'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse': 'drq_baseline',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue': 'doubleres_cap1_prefixed',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap4_nc32_addtrue': 'doubleres_cap4_prefixed',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue': 'doubleres_cap1_ul',
        'drq_double_n3_p0.001_encrl_pre100000_res_atc_cap1_nc32_addtrue_qinultrue': 'doubleres_cap1_qsig',
        'drq_double_n3_p0.001_encboth_pre100000_res_atc_cap1_nc32_addtrue_qinultrue': 'doubleres_cap1_ul_qsig',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_nc64_addfalse':'drq_base_pre100K_nc64'
    }

    for env in [
        # "acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy",
        # "finger_turn_hard", "hopper_hop", "quadruped_run", "quadruped_walk",
        # "reach_duplo", "reacher_easy", "reacher_hard", "walker_run",
        # "humanoid_walk",
        "acrobot_swingup", "finger_turn_easy", "finger_turn_hard", "quadruped_run", "reacher_easy", "reacher_hard",
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_medium6task_3seed_baseline_variants():
    list_of_data_dir = ['newbaseline3s', 'newdouble3s']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'compare3seed')
    save_prefix = 'compare3seed_baseline_variants'

    to_plot = [
        'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
        'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc64_addfalse',
        'drq_single_n3_p0.001_encrl_pre100000_res_atc_cap0_nc64_addfalse',
        'drq_single_n3_p0.001_encrl_pre100000_res_curl_cap0_nc64_addfalse',
        'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc64_addfalse'
    ]

    for env in [
        # "acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy",
        # "finger_turn_hard", "hopper_hop", "quadruped_run", "quadruped_walk",
        # "reach_duplo", "reacher_easy", "reacher_hard", "walker_run",
        # "humanoid_walk",
        "acrobot_swingup", "finger_turn_easy", "finger_turn_hard", "quadruped_run", "reacher_easy", "reacher_hard",
    ]:
        print(env)
        for y in ['performance',
                  # 'pretanh', 'critic_loss', 'critic_q1'
                  ]:
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 # variant2colors=expand_with_env_name(variant2colors, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 # variant2labels=expand_with_env_name(variant2labels, env),
                 smooth=10,
                 save_folder=save_folder,
                 save_name=save_name
                 )

def plot_pt_rrl():
    list_of_data_dir = ['pt', 'newbaseline3s',]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pt')
    envs = ["cheetah_run"]

    save_prefix_list = ['pt_rrl_add0', 'pt_rrl_add1', 'pt_rrl_add2']
    to_plot_list = [
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            'trans_resnet18true_nft0_add0',
        'trans_resnet18true_nft1_add0',
        'trans_resnet18true_nft2_add0',
        'trans_resnet18true_nft3_add0',
        'trans_resnet18true_nft4_add0',
    ],
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            'trans_resnet18true_nft0_add1',
        'trans_resnet18true_nft1_add1',
        'trans_resnet18true_nft2_add1',
        'trans_resnet18true_nft3_add1',
        'trans_resnet18true_nft4_add1',
    ],
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            'trans_resnet18true_nft0_add2',
        'trans_resnet18true_nft1_add2',
        'trans_resnet18true_nft2_add2',
        'trans_resnet18true_nft3_add2',
        'trans_resnet18true_nft4_add2',
    ],
    ]
    for i in range(3):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )

def plot_utd():
    list_of_data_dir = ['utd']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'utd')
    envs = ["acrobot_swingup", "finger_turn_easy", "finger_turn_hard", "quadruped_run", "reacher_easy", "reacher_hard"]

    save_prefix_list = ['utd']
    to_plot_list = [
        [
        'drq_utd1',
        'drq_utd3',
        'drq_utd5',
        'drq_utd10',
        ],
    ]
    for i in range(len(save_prefix_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )

def plot_stage3_conv2():
    list_of_data_dir = ['pt', 'newbaseline3s',]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pt')
    envs = ["acrobot_swingup", "finger_turn_easy", "finger_turn_hard", "quadruped_run", "reacher_easy", "reacher_hard",]

    save_prefix_list = ['stage3_conv2']
    to_plot_list = [
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            'drq_conv2_resnet18_32channel_ftnone',
            'drq_conv2_resnet18_32channel_fthalf',
            'drq_conv2_resnet18_32channel_ftall',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )


def plot_stage3_conv2_debug():
    list_of_data_dir = ['pt', 'newbaseline3s',]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pt')
    envs = ["acrobot_swingup", "finger_turn_hard", "quadruped_run", "reacher_hard",]

    save_prefix_list = ['stage3_conv2debug']
    to_plot_list = [
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            # 'drq_conv2_resnet18_32channel_ftall',
            'drq_none_ftall', # drq with data normalization
            'drq_none_ftnone', # drq with fixed encoder, data normalization
            'drq_none_ftnone_augfalse',
            'drq_conv2_resnet18_32channel_ftnone_augfalse',
            'drq_none_ftall_augfalse',
            'drq_conv2_resnet18_32channel_ftall_augfalse',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )


def plot_stage2_conv4():
    list_of_data_dir = ['pt', 'newbaseline3s',]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'pt')
    envs = [
        # "cartpole_swingup",
        #     "cheetah_run",
        # "finger_turn_hard",
        # "quadruped_walk",
        # "hopper_hop",
        "walker_walk"
    ]

    save_prefix_list = ['stage2test']
    to_plot_list = [
        [
            'drq_single_n3_p0.001_encrl_pre0_res_atc_cap0_nc32_addfalse',
            # 'drq_none_ftall',  # drq with data normalization
            # 'drq_none_ftnone',  # drq with fixed encoder, data normalization
            # 'drq_none_stage2200000_ftnone_augtrue',
            'drq_none_stage2200000_ftall_augtrue',
            # 'drq_conv4_resnet18_32channel_stage2200000_ftnone_augtrue',
            'drq_conv4_resnet18_32channel_stage2200000_ftall_augtrue',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )

def plot_double_res_human():
    list_of_data_dir = ['newdouble3s',]
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'hum')
    envs = [
        "humanoid_run", "humanoid_stand",
    ]

    save_prefix_list = ['doubleres']
    to_plot_list = [
        [
            # 'drq_double_n3_p0.001_encrl_pre10000_res_atc_cap0_nc32_addtrue_qinultrue',
            # 'drq_double_n3_p0.001_encrl_pre10000_res_atc_cap1_nc32_addtrue_qinultrue',
            # 'drq_double_n3_p0.001_encboth_pre10000_res_atc_cap0_nc32_addtrue_qinultrue',
            # 'drq_double_n3_p0.001_encboth_pre10000_res_atc_cap1_nc32_addtrue_qinultrue',
            'drq_double_n3_p0.001_encrl_pre10000_res_atc_cap0_nc32_addfalse_qinultrue',
            'drq_double_n3_p0.001_encrl_pre10000_res_atc_cap1_nc32_addfalse_qinultrue',
            'drq_double_n3_p0.001_encboth_pre10000_res_atc_cap0_nc32_addfalse_qinultrue',
            'drq_double_n3_p0.001_encboth_pre10000_res_atc_cap1_nc32_addfalse_qinultrue',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )

def plot_ssl_1():
    list_of_data_dir = ['drq-base', 'ssl', 'tl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ssl')
    envs = [
        "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    save_prefix_list = ['ssl-stage1-stage3-freeze',
                        'ssl-stage1-stage3-ft-32ch',
                        'ssl-stage1-stage3-ft-64ch',
                        'ssl-stage1-stage3-ft-96ch',
                        'ssl-stage1-stage3-ft-128ch',
                        ]
    to_plot_list = [
        [
            'drq_none_ftall', # drq baseline
            'drq_none_ftnone',  # random fixed baseline
            'tl_conv4_resnet18_64channel_ftnone', # tl fixed baseline
            'ssl_conv4_32channel_ftnone', # ssl fixed baseline
            'ssl_conv4_64channel_ftnone',  # ssl fixed baseline
            'ssl_conv4_96channel_ftnone',  # ssl fixed baseline
            'ssl_conv4_128channel_ftnone',  # ssl fixed baseline
        ],
        [
            'drq_none_ftall',  # drq baseline
            'ssl_conv4_32channel_ftall_elr1e-06',
            'ssl_conv4_32channel_ftall_elr1e-05',
            'ssl_conv4_32channel_ftall_elr0.0001',
            'ssl_conv4_32channel_ftall_elr0.001',
            'ssl_conv4_32channel_ftall_elr0.01',
        ],
        [
            'drq_none_ftall',  # drq baseline
            'ssl_conv4_64channel_ftall_elr1e-06',
            'ssl_conv4_64channel_ftall_elr1e-05',
            'ssl_conv4_64channel_ftall_elr0.0001',
            'ssl_conv4_64channel_ftall_elr0.001',
            'ssl_conv4_64channel_ftall_elr0.01',
        ],
        [
            'drq_none_ftall',  # drq baseline
            'ssl_conv4_96channel_ftall_elr1e-06',
            'ssl_conv4_96channel_ftall_elr1e-05',
            'ssl_conv4_96channel_ftall_elr0.0001',
            'ssl_conv4_96channel_ftall_elr0.001',
            'ssl_conv4_96channel_ftall_elr0.01',
        ],
        [
            'drq_none_ftall',  # drq baseline
            'ssl_conv4_128channel_ftall_elr1e-06',
            'ssl_conv4_128channel_ftall_elr1e-05',
            'ssl_conv4_128channel_ftall_elr0.0001',
            'ssl_conv4_128channel_ftall_elr0.001',
            'ssl_conv4_128channel_ftall_elr0.01',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_folder,
                     save_name=save_name
                     )

def plot_ssl_1_aggregate():
    list_of_data_dir = ['drq-base', 'ssl', 'tl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ssl')
    envs = [
        "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    # each curve is a learning rate setting, aggregate over 4 architectures and 6 environments.
    label2variants = {
        'baseline':['drq_none_ftall'],
        'tl':['tl_conv4_resnet18_32channel_ftnone'],
        'ssl-freeze':['ssl_conv4_32channel_ftnone', 'ssl_conv4_64channel_ftnone', 'ssl_conv4_96channel_ftnone', 'ssl_conv4_128channel_ftnone', ],
        '1e-2': ['ssl_conv4_32channel_ftall_elr0.01','ssl_conv4_64channel_ftall_elr0.01','ssl_conv4_96channel_ftall_elr0.01','ssl_conv4_128channel_ftall_elr0.01', ],
        '1e-3': ['ssl_conv4_32channel_ftall_elr0.001','ssl_conv4_64channel_ftall_elr0.001','ssl_conv4_96channel_ftall_elr0.001','ssl_conv4_128channel_ftall_elr0.001', ],
        '1e-4': ['ssl_conv4_32channel_ftall_elr0.0001','ssl_conv4_64channel_ftall_elr0.0001','ssl_conv4_96channel_ftall_elr0.0001','ssl_conv4_128channel_ftall_elr0.0001', ],
        '1e-5': ['ssl_conv4_32channel_ftall_elr1e-05','ssl_conv4_64channel_ftall_elr1e-05','ssl_conv4_96channel_ftall_elr1e-05','ssl_conv4_128channel_ftall_elr1e-05', ],
        '1e-6': ['ssl_conv4_32channel_ftall_elr1e-06','ssl_conv4_64channel_ftall_elr1e-06','ssl_conv4_96channel_ftall_elr1e-06','ssl_conv4_128channel_ftall_elr1e-06', ],
    }

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    save_prefix = 'ssl-lr-aggregate-easy6'
    y_to_plot = 'performance'
    save_name = '%s_%s' % (save_prefix, y_to_plot)
    plot_aggregate(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name)


def plot_ssl_stage2():
    list_of_data_dir = ['drq-base', 'ssl', 'tl', 's2-base', 'ssl-s2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ssl-stage2')
    envs = [
        "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    save_prefix_list = ['ssl-stage1-stage2-stage3',
                        'ssl-stage1-stage2-stage3-ft-32ch',
                        'ssl-stage1-stage2-stage3-ft-64ch',
                        'ssl-stage1-stage2-stage3-ft-96ch',
                        ]
    to_plot_list = [
        [
            'drq_none_ftall', # drq baseline
            'drq_none_nc32_stage2_100000_ftnone',# s2 only - freeze
            'drq_none_nc32_stage2_100000_ftall', # s2 only -finetune
            'drq_conv4_32channel_stage2_100000_ftnone', # s1-s2-freeze
            'drq_conv4_32channel_stage2_100000_ftall_elr0.0001',  # s1-s2 finetune, same lr
        ],
        [
            'drq_none_ftall',  # drq baseline
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_32channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_32channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_32channel_stage2_100000_ftall_elr0.001',
        ],
        [
            'drq_none_ftall',  # drq baseline
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_64channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_64channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_64channel_stage2_100000_ftall_elr0.001',
        ],
        [
            'drq_none_ftall',  # drq baseline
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_96channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_96channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_96channel_stage2_100000_ftall_elr0.001',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['performance',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )


def plot_ssl_stage2_extra():
    list_of_data_dir = ['drq-base', 'ssl', 'tl', 's2-base', 'ssl-s2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ssl-stage2-extra')
    envs = [
        "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    save_prefix_list = ['ssl-stage1-stage2-stage3',
                        'ssl-stage1-stage2-stage3-ft-32ch',
                        'ssl-stage1-stage2-stage3-ft-64ch',
                        'ssl-stage1-stage2-stage3-ft-96ch',
                        ]
    to_plot_list = [
        [
            'drq_none_nc32_stage2_100000_ftnone',# s2 only - freeze
            'drq_none_nc32_stage2_100000_ftall', # s2 only -finetune
            'drq_conv4_32channel_stage2_100000_ftnone', # s1-s2-freeze
            'drq_conv4_32channel_stage2_100000_ftall_elr0.0001',  # s1-s2 finetune, same lr
        ],
        [
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_32channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_32channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_32channel_stage2_100000_ftall_elr0.001',
        ],
        [
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_64channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_64channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_64channel_stage2_100000_ftall_elr0.001',
        ],
        [
            'drq_none_nc32_stage2_100000_ftall',  # s2 only -finetune
            'drq_conv4_96channel_stage2_100000_ftall_elr1e-05',
            'drq_conv4_96channel_stage2_100000_ftall_elr0.0001',
            'drq_conv4_96channel_stage2_100000_ftall_elr0.001',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in ['critic_q1', 'critic_loss', 'actor_logprob'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_ssl_stage2_aggregate():
    list_of_data_dir = ['drq-base', 'ssl', 'tl', 's2-base', 'ssl-s2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ssl-stage2-aggregate')
    envs = [
        "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    # each curve is a learning rate setting, aggregate over 4 architectures and 6 environments.
    label2variants = {
        'baseline':['drq_none_ftall'],
        'stage2-1e-4': ['drq_none_nc32_stage2_100000_ftall','drq_none_nc64_stage2_100000_ftall','drq_none_nc96_stage2_100000_ftall',],
        'stage2-freeze': ['drq_none_nc32_stage2_100000_ftnone','drq_none_nc64_stage2_100000_ftnone','drq_none_nc96_stage2_100000_ftnone',],
        'ssl-nostage2-freeze':['ssl_conv4_32channel_ftnone', 'ssl_conv4_64channel_ftnone', 'ssl_conv4_96channel_ftnone', 'ssl_conv4_128channel_ftnone', ],
        'ssl-nostage2-1e-4': ['ssl_conv4_32channel_ftall_elr0.0001','ssl_conv4_64channel_ftall_elr0.0001','ssl_conv4_96channel_ftall_elr0.0001','ssl_conv4_128channel_ftall_elr0.0001', ],
        'ssl-stage2-1e-4': ['drq_conv4_32channel_stage2_100000_ftall_elr0.0001',
                            'drq_conv4_64channel_stage2_100000_ftall_elr0.0001',
                            'drq_conv4_96channel_stage2_100000_ftall_elr0.0001', ],
        'ssl-stage2-1e-3': ['drq_conv4_32channel_stage2_100000_ftall_elr0.001',
                            'drq_conv4_64channel_stage2_100000_ftall_elr0.001',
                            'drq_conv4_96channel_stage2_100000_ftall_elr0.001', ],
    }

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    save_prefix = 'ssl-stage2-lr-aggregate-easy6'
    y_to_plot = 'performance'
    save_name = '%s_%s' % (save_prefix, y_to_plot)
    plot_aggregate(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name)



def plot_atest():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2')
    envs = [
        "pen", "hammer", "door", "relocate"
    ]

    save_prefix_list = ['ad2_none',
                        'ad2_ssl',
                        'ad2_tl'
                        ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            # 'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance','critic_q1', 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_atest2_success_rate():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-compare')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = ['ad2_none',
                        'ad2_ssl',
                        'ad2_tl_old'
                        ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_atest2_success_rate_best():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-compare')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = ['ad2_best',
                        ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_atest2_no_data_aug():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-no-aug')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = ['ad2_no_aug',
                        ]
    to_plot_list = [
        [
            # 'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            # 'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            # 'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d25_augFalse',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d25_augFalse',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d25_augFalse',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                'critic_q1',
                'critic_loss'
            ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                if y in ('critic_q1', 'critic_loss'):
                    skip = 40
                else:
                    skip = 1
                if y == 'critic_loss':
                    y_log_scale = True
                else:
                    y_log_scale = False

                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip,
                     y_log_scale=y_log_scale
                     )

def plot_atest2_small_lr_old():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-small-lr-old')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = [
        'ad2_small_lr_ssl',
        'ad2_small_lr_tl',
        'ad2_small_lr_none',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_conv4_32channel_lr1e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_32channel_lr3e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_32channel_lr3e-06_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr1e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr3e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr3e-06_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_none_lr1e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr3e-05_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr3e-06_es0.01_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ], # 82.48 / 100 -> 83.18 / 100
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 40 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )

def plot_atest2_small_lr():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-small-lr')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = [
        'ad2_small_lr_ssl',
        'ad2_small_lr_tl',
        'ad2_small_lr_none',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.003_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.0001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.003_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.0001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.003_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr0.0001_es0.001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr0.0001_es0.0001_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue',
            'approx_rrl_baseline'
        ], # 82.48 / 100 -> 83.18 / 100
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 40 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )

def plot_drq_with_rrl_resnet_feature():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-rrl-feature')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = [
        'ad2_resnet18',
        'ad2_resnet34',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_256',
            'adroit_trial_ar2_none_lr3e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_256',
            'adroit_trial_ar2_none_lr1e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_256',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_1024',
            'adroit_trial_ar2_none_lr3e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_1024',
            'adroit_trial_ar2_none_lr1e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet18_1024',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_256',
            'adroit_trial_ar2_none_lr3e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_256',
            'adroit_trial_ar2_none_lr1e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_256',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_1024',
            'adroit_trial_ar2_none_lr3e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_1024',
            'adroit_trial_ar2_none_lr1e-05_es0_bc0.1_0.95_sa1000_ra0_d25_augFalse_resnet34_1024',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                # 'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                if y in  ('critic_q1', 'critic_loss'):
                    skip = 40
                else:
                    skip = 1
                if y == 'critic_loss':
                    y_log_scale = True
                else:
                    y_log_scale = False

                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip,
                     y_log_scale=y_log_scale
                     )

def plot_adroit_stable_success_rate(): # plot the new drq variants in adroit that uses safe q threshold and other modifications
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable')
    envs = [
        # "pen",
        "hammer",
        # "door",
        # "relocate"
    ]

    save_prefix_list = [
        'adstable_none',
        'adstable_ssl',
        'adstable_tl',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                if y in  ('critic_q1', 'critic_loss'):
                    skip = 40
                else:
                    skip = 1
                if y == 'critic_loss':
                    y_log_scale = True
                else:
                    y_log_scale = False

                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip,
                     y_log_scale=y_log_scale
                     )

def plot_adroit_stable_distraction_success_rate(): # plot the new drq variants in adroit that uses safe q threshold and other modifications
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable')
    envs = [
        # "pen",
        "hammer",
        # "door",
        # "relocate"
    ]

    save_prefix_list = [
        'adstable_none',
        'adstable_ssl',
        'adstable_tl',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
        ],
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d5_augTrue_stds0.2',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0_d15_augTrue_stds0.2',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate_aug',
                # 'performance',
                'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                if y in  ('critic_q1', 'critic_loss'):
                    skip = 40
                else:
                    skip = 1
                if y == 'critic_loss':
                    y_log_scale = True
                else:
                    y_log_scale = False

                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip,
                     y_log_scale=y_log_scale
                     )

def plot_atest2_others():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-compare')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = ['ad2_none',
                        'ad2_ssl',
                        'ad2_tl_old'
                        ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
        [
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )


def plot_atest2_aggregate():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-compare', 'aggregate')
    envs = [
        "pen", "hammer",
        "door",
        # "relocate"
    ]

    # each curve is a learning rate setting, aggregate over 4 architectures and 6 environments.
    label2variants = {
        'baseline':['approx_rrl_baseline'],
        'none': ['adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',],
        'ssl': ['adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',],
        'tl':['adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.1_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',],
    }

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    save_prefix = 'door-pen-hammer-aggregate'
    y_to_plot = 'success_rate'
    save_name = '%s_%s' % (save_prefix, y_to_plot)
    plot_aggregate(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name, plot_y_value=y_to_plot)


def plot_atest3():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-compare')
    envs = [
        "door",
        "pen", "hammer",
        # "relocate"
    ]

    save_prefix_list = [
        # 'ad2_none',
        # 'ad2_ssl',
        'ad2_tl'
    ]
    to_plot_list = [
        # [
        #     'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
        #     'adroit_trial_ar2_none_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
        #     'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        #     # 'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
        #     # 'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
        #     # 'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
        #     # 'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
        #     # 'approx_rrl_baseline'
        #     # 'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa1000_ra0',
        # ],
        # [
        #     'adroit_trial_ar2_conv4_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
        #     'adroit_trial_ar2_conv4_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
        # ],
        [
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es0.01_bc0.01_0.95_sa1000_ra0',
            'adroit_trial_ar2_conv4_resnet18_32channel_lr0.0001_es1_bc0.01_0.95_sa1000_ra0',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_atest_stable_small_policy_lr():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-policy-lr']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable-policy-lr')
    envs = [
        # "door",
        "pen", "hammer",
        # "relocate"
    ]

    save_prefix_list = [
        'ad2_none',
        'ad2_none_bc1',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als1',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.1',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.01',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.001',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als1',
            'adroit_trial_ar2_none_lr0.0001_es1_bc1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.1',
            'adroit_trial_ar2_none_lr0.0001_es1_bc1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.01',
            'adroit_trial_ar2_none_lr0.0001_es1_bc1_0.95_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min30_als0.001',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )


def check_adroit_first_group():
    dir = os.path.join(base_dir, 'adroit-firstgroup')
    task = 'pen'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=1500000, end_frame=2000000, eval_criteria=eval_criteria)

def check_adroit_stable_group():
    dir = os.path.join(base_dir, 'adroit-stable-hyper')
    task = 'hammer'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=1200000, end_frame=1500000, eval_criteria=eval_criteria)

def check_adroit_stable_group2():
    dir = os.path.join(base_dir, 'adroit-stable-hyper2')
    task = 'pen'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=1500000, end_frame=2000000, eval_criteria=eval_criteria)

def check_adroit_cql():
    dir = os.path.join(base_dir, 'adroit-s2cql')
    task = 'door'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=0, end_frame=100000, eval_criteria=eval_criteria)

def check_relocate():
    dir = os.path.join(base_dir, 'adroit-relocatehyper4')
    dir = os.path.join(base_dir, 'adroit-relocatehyper3')
    dir = os.path.join(base_dir, 'adroit-relocatehyper2')
    dir = os.path.join(base_dir, 'adroit-relocatehyper3-best-10seed')

    # dir = os.path.join(base_dir, 'adroit-relocatehyper') # this is with rrl features
    task = 'relocate'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=200000, end_frame=500000, eval_criteria=eval_criteria)

def check_test():
    dir = os.path.join(base_dir, 'adroit-s2cql')
    # dir = os.path.join(base_dir, 'adroit-relocatehyper3')
    # dir = os.path.join(base_dir, 'adroit-relocatehyper') # this is with rrl features
    task = 'relocate'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=200000, end_frame=500000, eval_criteria=eval_criteria)


def check_res6pen():
    dir = os.path.join(base_dir, 'res6-pen')
    # dir = os.path.join(base_dir, 'adroit-relocatehyper') # this is with rrl features
    task = 'pen'
    eval_criteria = 'success_rate'
    hyper_search_adroit([dir], task_name=task, start_frame=0, end_frame=1000000, eval_criteria=eval_criteria)

def plot_10seed_relocate():
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-relocate10s')
    envs = [
        # "door",
        # "pen", "hammer",
        "relocate"
    ]

    save_prefix_list = [
        'a',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_cql_best_hammer():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable-hyper']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable-hyper')
    envs = [
        # "pen",
        "hammer",
        # "door",
        # "relocate"
    ]

    save_prefix_list = [
        'hyper_best',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0002_es0_bc0.001_0.95_sa1000_ra0_d25_augTrue_stds0.2_qt200_qf0',
            'adroit_trial_ar2_none_lr0.0002_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue_stds0.1_qt125_qf0.2',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.9_sa2000_ra0_d25_augTrue_stds0.2_qt125_qf0.9',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.01_0.95_sa1000_ra0_d25_augTrue_stds0.5_qt150_qf0',
            'adroit_trial_ar2_conv4_32channel_lr0.0005_es0_bc0.001_0.95_sa1000_ra0_d25_augTrue_stds0.5_qt200_qf0.5',
            # 'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )


def plot_atest2_stable_hyper_door():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable-hyper']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable-hyper')
    envs = [
        # "pen",
        # "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = [
        'hyper_best',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.05_0.9_sa2000_ra0_d25_augTrue_stds0.2_qt150_qf0',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.05_0.9_sa1000_ra0_d25_augTrue_stds0.5_qt125_qf0.9',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.001_0.95_sa5000_ra0_d25_augTrue_stds0.2_qt125_qf0',
            'adroit_trial_ar2_conv4_32channel_lr0.0002_es0_bc0.1_0.95_sa2000_ra0_d25_augTrue_stds1.0_qt100_qf0.9',
            'adroit_trial_ar2_conv4_32channel_lr0.0001_es0_bc0.05_0.9_sa1000_ra0_d25_augTrue_stds0.1_qt100_qf0.5',
            # 'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )

def plot_atest2_stable_hyper_hammer():
    list_of_data_dir = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable-hyper']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-stable-hyper')
    envs = [
        # "pen",
        "hammer",
        # "door",
        # "relocate"
    ]

    save_prefix_list = [
        'hyper_best',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0002_es0_bc0.001_0.95_sa1000_ra0_d25_augTrue_stds0.2_qt200_qf0',
            'adroit_trial_ar2_none_lr0.0002_es0_bc0.1_0.95_sa1000_ra0_d25_augTrue_stds0.1_qt125_qf0.2',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.1_0.9_sa2000_ra0_d25_augTrue_stds0.2_qt125_qf0.9',
            'adroit_trial_ar2_none_lr0.0001_es0_bc0.01_0.95_sa1000_ra0_d25_augTrue_stds0.5_qt150_qf0',
            'adroit_trial_ar2_conv4_32channel_lr0.0005_es0_bc0.001_0.95_sa1000_ra0_d25_augTrue_stds0.5_qt200_qf0.5',
            # 'approx_rrl_baseline' 'adroit-firstgroup',
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )


def plot_1231_hyper3():
    list_of_data_dir = ['adroit-s2cql2', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-cqlnew')
    envs = [
        "pen",
        "hammer",
        "door",
        # "relocate"
    ]

    save_prefix_list = [
        'cql-new',
        'cql-new-rrl',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.003_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.003_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'approx_rrl_baseline'
        ],
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.003_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augFalse_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64_resnet18',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augFalse_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64_resnet18',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.003_0.9_sa5000_ra0_d25_augFalse_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64_resnet18',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.003_0.9_sa5000_ra0_d25_augFalse_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64_resnet18',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )

def plot_1231_relocate():
    list_of_data_dir = ['adroit-s2cql2', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-cqlnew')
    envs = [
        "relocate"
    ]

    save_prefix_list = [
        'cql-new',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr0.0003_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr3e-05_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'adroit_trial_ar2_none_lr0.0003_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'adroit_trial_ar2_none_lr3e-05_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d10000_dr64',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )


def plot_1231_hyper6_stage2_up_encoder_only(): # init with random encoder, and update encoder in stage 2 only
    list_of_data_dir = ['adroit-s2cql2', 'adroit-secondgroup']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-cqlnew')
    envs = [
        "pen",
        "hammer",
        # "door",
        # "relocate"
    ]

    save_prefix_list = [
        'stage2-encoder-only',
    ]
    to_plot_list = [
        [
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64',
            'adroit_trial_ar2_none_lr0.0001_es1_bc0.001_0.9_sa5000_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_20000_d0_dr64_s3noeTrue',
            'approx_rrl_baseline'
        ],
    ]
    for i in range(len(to_plot_list)):
        save_prefix = save_prefix_list[i]
        to_plot = to_plot_list[i]
        for env in envs:
            print(env)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                skip = 5 if y in ('critic_loss', 'critic_q1') else 1
                y_log_scale = True if y == 'critic_loss' else False
                ymax = 100000 if y == 'critic_q1' else None
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     data_skip=skip, y_log_scale=y_log_scale, ymax=ymax
                     )

def plot_relocate_first_ablation(): # we will now plot ablation results we obtianed on Jan 10, 2021
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-relocate-abl')
    envs = [
        # "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'lr': # 1
        [
            'approx_rrl_baseline',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr3e-05_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr0.0003_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr0.001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
        ],
        'net': # 2
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'enclr':  # 3
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.0001_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.001_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.1_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es1_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'bc':  # 4
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.1_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.01_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.0001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'seed_action':  # 5
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa5000_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa20000_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'no_aug':  # 6
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augFalse_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0_bc0.001_0.5_sa0_ra0_d25_augFalse_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'std0':  # 7
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.05_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.1_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.5_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'cqlw':  # 8
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql0.001_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql0.01_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql0.1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql10_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'cql_update':  # 9
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_5000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_50000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_100000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        's2cloned_data':  # 10
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d30000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d100000_dr64_pixels_fs3_cb0_r500000',
            ],
        'cql_bc_weight':  # 11
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0.0001_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0.001_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0.01_r500000',
            ],
        # 'lr':
        #     [
        #         'approx_rrl_baseline',
        #         'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
        #         'adroit_trial_ar2_resnet6_32channel_lr3e-05_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
        #         'adroit_trial_ar2_resnet6_32channel_lr0.0003_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
        #         'adroit_trial_ar2_resnet6_32channel_lr0.001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
        #     ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_relocate_second_ablation(): # we will now plot ablation results we obtianed on Jan 11, 2021
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-relocate-abl')
    envs = [
        # "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'demo_batch': # 12
        [
            'approx_rrl_baseline',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr32_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr128_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr256_pixels_fs3_cb0_r500000',
        ],
        'net10':  # 13
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet10_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
            ],
        'cql_n_random':  # 14
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_10',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_20',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_50',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_100',
            ],
        'cql_std':  # 15
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_10',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_10_0.01',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_10_0.1',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d10000_dr64_pixels_fs3_cb0_r500000_10_0.2',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_abs2_more(): # we will now plot ablation results we obtianed on Jan 11, 2021
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'ad2-3envs-abl')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'net': # 1
        [
            'approx_rrl_baseline',
            'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_resnet10_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
            'adroit_trial_ar2_none_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
        ],
        'std':  # 1
            [
                'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.01_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.05_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.1_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.2_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.3_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000',
            ],
        'enc_lr_scale':  # 1
            [
                # 'approx_rrl_baseline',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.01_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.05_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000_False',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es0.1_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.05_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000_False',
                'adroit_trial_ar2_resnet6_32channel_lr0.0001_es1_bc0.001_0.5_sa0_ra0_d25_augTrue_stds0.05_qt200_qf0.5_dn3_0_min-50_als1_cql1_30000_d0_dr64_pixels_fs3_cb0_r500000_False',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    plot_this_time = ['enc_lr_scale']
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                # 'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                'abs_pretanh',
                'max_abs_pretanh',
                'actor_loss_bc'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_main(): # plot the main results we obtained from 1-12-2022
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2', 'main']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'adroit-main')
    envs = [
        "door",
        "pen", "hammer",
        # "relocate"
    ]

    prefix_to_list_of_variants = {
        'check': # 1
        [
            'approx_rrl_baseline',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.3_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.3_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
        ],
        'check2':  # 1
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.3_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
                'vrl3_ar2_fs3_pixels_none_pFalse_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.3_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                # 'actor_loss_bc'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_main_relocate(): # plot the main results we obtained from 1-12-2022
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2', 'main']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'adroit-main')
    envs = [
        # "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'check2':  # 1
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                # 'actor_loss_bc'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_main2(): # results got on 1-13-2022
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2', 'main2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'adroit-main2')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'ef0':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
        'ef1':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
            ],
        'ef2':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'performance',
                # 'critic_q1',
                # 'critic_loss',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                # 'actor_loss_bc'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_main2_analysis(): # results got on 1-13-2022
    list_of_data_dir = ['adroit-relocatehyper3-best-10seed', 'adroit-secondgroup', 'relocate-abl1', 'abl2', 'main2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'adroit-main2')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'ef0':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
        'ef1':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etFalse',
            ],
        'ef2':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                # 'success_rate',
                # 'performance',
                'critic_q1',
                'critic_loss',
                'abs_pretanh',
                'max_abs_pretanh',
                'actor_loss_bc'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_varl3_main_door_hammer_pen(): # paper result (need to replace RRL scores)
    list_of_data_dir = ['rrl-approx-base', 'vrl3-main']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'vrl3-main')
    envs = [
        "door",
        "pen", "hammer",
        # "relocate"
    ]

    prefix_to_list_of_variants = {
        'vrl3_main':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_varl3_main_relocate(): # paper result (need to replace RRL scores)
    list_of_data_dir = ['rrl-approx-base', 'vrl3-main']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'vrl3-main')
    envs = [
        # "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'vrl3_main':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_s2eTrue_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

def plot_main4(): # paper result (need to replace RRL scores)
    list_of_data_dir = ['rrl-approx-base', 'main4']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'main4')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'vrl3_main':
            [
                'approx_rrl_baseline',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     # variant2colors=expand_with_env_name(variant2colors, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     # variant2labels=expand_with_env_name(variant2labels, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name
                     )

NUM_MILLION_ON_X_AXIS = 4
DEFAULT_Y_MIN = -0.02
DEFAULT_X_MIN = -50000
DEFAULT_Y_MAX = 1.02
DEFAULT_X_MAX = 4050000
X_LONG = 12050000
DEFAULT_DMC_X_MAX = 3050000
DEFAULT_DMC_EASY_X_MAX = 1050000
DEFAULT_DMC_HARD_X_MAX = 30500000
ADROIT_ALL_ENVS =  ["door", "pen", "hammer","relocate"]
DMC_EASY_ENVS = ["cartpole_balance", "cartpole_balance_sparse", "cartpole_swingup", "cup_catch",
                 "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk",]
DMC_MEDIUM_ENVS = ["acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy", "finger_turn_hard",
        "hopper_hop", "quadruped_run", "quadruped_walk", "reach_duplo", "reacher_easy", "reacher_hard", "walker_run",]
DMC_HARD_ENVS = ["humanoid_stand", "humanoid_walk", "humanoid_run"]
DMC_ALL_ENVS = DMC_EASY_ENVS + DMC_MEDIUM_ENVS + DMC_HARD_ENVS

def plot_paper_main():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'main')
    envs = [
        "door",
        "pen",
        "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'rrl':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'rrl': 'RRL'
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'rrl': 'tab:grey',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                # 'success_rate',
                'success_rate_aug'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = 0
                    ymax = 1
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax, xmax=int(1e6) * 4,
                     no_legend=False if env=='relocate' else True,
                     )

def plot_paper_main_more():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'main')
    envs = [
        "door",
        "pen",
        "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'rrl_more':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'rrl',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'rrl': 'RRL',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue':'DrQv2fD(VRL3)',
         'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000':'FERM(VRL3)',
          'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2contrast100000':'VRL FERM',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'rrl': 'tab:grey',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000': 'tab:blue',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2contrast100000': 'tab:purple',
    }

    variant2dashes = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'rrl': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000': True,
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'success_rate_aug'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax, xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=False if env=='relocate' else True,
                     max_seed=11,
                     )

# one option is to set a parameter: list of envs. All envs will be aggregated. If just one env is given, then we do the
# per-env plot
def decide_placeholder_prefix(envs, folder_name, plot_name):
    if isinstance(envs, list):
        if len(envs) > 1:
            return 'aggregate-' + folder_name, 'aggregate_' + plot_name
        else:
            return folder_name,plot_name +'_'+ envs[0]
    return folder_name, plot_name +'_'+ envs

# add sth here so that if we plot per-task, then it's saved to a different folder
# and should figure out a way to decide which plots to add labels...
def plot_paper_main_more_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'adroit_main')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'RRL', 'DrQv2fD(VRL3)', 'FERM(VRL3)', 'FERM+S1', 'VRL3+FERM']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['rrl'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2contrast30000'],
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


def plot_nrebuttal_long_term(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'adroit_main_long')
    list_of_data_dir = ['abl', 'ablph']
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':X_LONG} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)
def plot_nrebuttal_long_term_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_nrebuttal_long_term(envs=[env], no_legend=(env!='relocate'))

def plot_nrebuttal_bc_new_abl(envs=ADROIT_ALL_ENVS, no_legend=False, fs1=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'bc_new_abl')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'BC Offline Act', 'BC MC', 'BC Naive']
    if fs1:
        variants = [
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n3'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n100'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ]
    else:
        variants = [
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n3'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n100'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)


def plot_nrebuttal_bc_new_abl_nos1(envs=ADROIT_ALL_ENVS, no_legend=False, fs1=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'bc_new_abl_nos1')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'BC Offline Act', 'BC MC', 'BC Naive']
    if fs1:
        variants = [
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n3'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n100'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ]
    else:
        variants = [
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n3'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2v30000_s2n100'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)

def plot_nrebuttal_bc_new_abl_per_task():
    for env in ['door', 'hammer', 'pen', 'relocate']: # , 'relocate'
        fs1 = True if env == 'relocate' else False
        plot_nrebuttal_bc_new_abl(envs=[env], no_legend=(env!='relocate'), fs1=fs1)

def plot_nrebuttal_bc_new_abl_nos1_per_task():
    for env in ['door', 'hammer', 'pen', 'relocate']: # , 'relocate'
        fs1 = True if env == 'relocate' else False
        plot_nrebuttal_bc_new_abl_nos1(envs=[env], no_legend=(env!='relocate'), fs1=fs1)


def plot_nrebuttal_s1_ssl(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 's1_ssl')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    # labels = [FRAMEWORK_NAME, 'BYOL', 'BYOL-shift', 'BYOL-all', 'No S1'] #
    labels = [FRAMEWORK_NAME, 'Contrast', 'Contrastive-shift', 'Contrastive-all', 'No S1'] #
    variants = [
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_byol_resnet6_32channel_e60_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_byol-shift_resnet6_32channel_e60_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_byol-all_resnet6_32channel_e60_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)

def plot_nrebuttal_highcap_10(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'highcap_res10')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    # 0, 0.001, 0.01, 0.1, 1
    labels = [FRAMEWORK_NAME,'Res10_0', 'Res10_0.001', 'Res10_0.01', 'Res10_0.1', 'Res10_1'] #
    variants = [
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)

def plot_nrebuttal_highcap_18(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'highcap_res18')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    # 0, 0.001, 0.01, 0.1, 1
    labels = [FRAMEWORK_NAME,'Res18_0', 'Res18_0.001', 'Res18_0.01', 'Res18_0.1', 'Res18_1'] #
    variants = [
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
        ['vrl3_ar2_fs1_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etFalse'],
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
        d = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX} if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d)

def plot_paper_main_relocate():
    save_prefix = 'just_relocate'
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'aggregate-fs')
    envs = ["relocate"]

    labels = [FRAMEWORK_NAME, 'RRL', 'DrQv2fD(VRL3)', 'FERM(VRL3)', 'FERM+S1', 'VRL3+FERM']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['rrl'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue_s2contrast30000'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s2contrast30000'],
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
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, **d)


def plot_paper_fs_s1_pretrain():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s1_pretrain':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'No S1 pretrain'
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:grey',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = 0
                    ymax = 1
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax, xmax=int(1e6) * 3,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s23_q_safe_factor():
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_q_safe_factor':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue':'No safe Q'
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue': 'tab:grey',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s23_q_safe_factor_aggregate():
    save_prefix = 'aggregate_fs_s23_q_safe_factor'
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'aggregate-fs')
    envs = ["door", "pen", "hammer","relocate"]

    labels = [FRAMEWORK_NAME, 'No safe Q']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:gray']

    label2variants, label2colors, label2dashes = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate', 'critic_loss', 'critic_q1']:
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, **d)


def plot_paper_fs_s23_transition(): # vrl3, stage 2 naive, stage 3 conservative
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_transition':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3cTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
                     'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue':'S2 Naive',
                     'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3cTrue':'S3 Cons',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue':'tab:gray',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3cTrue': 'tab:brown',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s23_transition_aggregate():
    save_prefix = 'aggregate_fs_s23_transition'
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'aggregate-fs')
    envs = ["door", "pen", "hammer","relocate"]

    labels = [FRAMEWORK_NAME, 'S2 Naive', 'S3 Cons',]
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3cTrue']
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown']

    label2variants, label2colors, label2dashes = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate', 'critic_loss', 'critic_q1']:
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, **d)


def plot_paper_fs_s23_transition_q_safe_factor_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s23_transition_q_safe_factor')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'S2 Naive', 'S2 Naive Safe', 'S3 Cons', 'No safe Q']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue', ],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3cTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown', 'tab:blue', 'tab:purple']
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

    for plot_y_value in ['success_rate', 'critic_q1']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        if plot_y_value == 'critic_q1':
            d['ymax'] = 1050
            d['ymin'] = -1050
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)
    ##################################

    # {'ymin': None, 'ymax': None, 'xmin': DEFAULT_X_MIN, 'xmax': DEFAULT_X_MAX}
    # for plot_y_value in ['success_rate', 'critic_q1']: #'critic_loss',
    #     d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
    #     d = copy.deepcopy(d)
    #     if plot_y_value == 'critic_q1':
    #         d['ymax'] = 1050
    #         d['ymin'] = -1050
    #     save_name = '%s_%s' % (save_prefix, plot_y_value)
    #     plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name,
    #                         plot_y_value=plot_y_value, label2colors=label2colors, **d)

def plot_paper_fs_s23_transition_q_safe_factor_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s23_transition_q_safe_factor_aggregate(envs=[env], no_legend=(env!='relocate'))

def plot_paper_fs_s23_enc_lr_scale(): # vrl3, stage 2 naive, stage 3 conservative
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_enc_lr_scale':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'0',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'0.001',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': '0.01',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'0.1',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': '1',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:brown',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:gray',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:blue',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN,
                     xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s23_enc_lr_scale_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s23_enc_lr_scale')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['0', '0.001', '0.01', '0.1', '1',]
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:brown', 'tab:gray', 'tab:red', 'tab:orange', 'tab:blue']
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

    for plot_y_value in [
        # 'success_rate', 'critic_q1',
                         'success_rate_aug'
    ]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value in ['success_rate', 'success_rate_aug'] else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)

def plot_paper_fs_s23_enc_lr_scale_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s23_enc_lr_scale_aggregate(envs=[env], no_legend=(env!='relocate'))

def plot_paper_fs_bn(): # vrl3, stage 2 naive, stage 3 conservative
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_bn':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode1',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode2',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode3',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'No Grad Eval',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode1': 'No Grad Train',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode2':'Grad Eval',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode3':'Grad Train',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode1': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode2':'tab:blue',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode3':'tab:gray',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                # 'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN,
                     xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s2_bc():
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s2_bc':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                #
                # 'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                #
                # 'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'Stage 2 BC',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'Stage 2 disabled',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:gray',
    }

    variant2dashes = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':  False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': True,
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin =DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s2_bc_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s2_bc')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [FRAMEWORK_NAME, 'Stage 2 BC', 'Stage 2 disabled', ]
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:orange', 'tab:gray', ]
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
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)

def plot_paper_fs_s2_bc_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s2_bc_aggregate(envs=[env], no_legend=(env!='relocate'))


def plot_paper_fs_s23_encoder_finetune():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_encoder_finetune':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': FRAMEWORK_NAME,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue': 'S2 update, S3 freeze',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S2 S3 freeze',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:gray',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = -0.01
                    ymax = 1
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax, xmax=int(1e6) * NUM_MILLION_ON_X_AXIS,
                     no_legend=no_legend,
                     )

def plot_paper_fs_s123_encoder_update():
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s123_encoder_finetune':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S123',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S1',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S23',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue': 'S12',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:gray',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue': 'tab:brown',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

adroit_success_rate_default_min_max_dict = {'ymin':DEFAULT_Y_MIN, 'ymax':DEFAULT_Y_MAX, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX}
adroit_other_value_default_min_max_dict = {'ymin':None, 'ymax':None, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_X_MAX}
dmc_all_default_min_max_dict = {'ymin':None, 'ymax':None, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_DMC_X_MAX}
dmc_hard_default_min_max_dict = {'ymin':None, 'ymax':None, 'xmin':DEFAULT_X_MIN, 'xmax':DEFAULT_DMC_HARD_X_MAX}

def plot_paper_fs_s123_encoder_update_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s123_encoder_finetune')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['S123', 'S1', 'S12', 'S2', 'S23', 'S3']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feFalse'],
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:blue', ]
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

    for plot_y_value in ['success_rate', 'critic_q1']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)
def plot_paper_fs_s123_encoder_update_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s123_encoder_update_aggregate([env], no_legend=(env!='relocate'))


def plot_paper_fs_s123_encoder_update_aggregate_extra(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s123_encoder_finetune')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['S123', 'S1', 'S12', 'S2', 'S23', 'S3']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_s3feFalse'],
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:blue', ]
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

    for plot_y_value in ['success_rate_aug']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if (plot_y_value == 'success_rate' or plot_y_value == 'success_rate_aug') else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)

def plot_paper_fs_s123_encoder_per_task_extra():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s123_encoder_update_aggregate_extra([env], no_legend=(env!='relocate'))


def plot_paper_fs_s123_stage_update_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False): # will plot VRL3 with certain stages disabled, S123, S23, S13, S3
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_s123_stage_update')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['S123', 'S23', 'S13', 'S3']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:orange', 'tab:pink', 'tab:blue']
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

    for plot_y_value in ['success_rate', 'critic_q1']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)
def plot_paper_fs_s123_stage_update_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_s123_stage_update_aggregate([env], no_legend=(env!='relocate'))

def plot_paper_fs_bn_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_bn')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['Eval/No Grad', 'Train/No Grad', 'Eval/Grad', 'Train/Grad']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode1'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode2'],
    ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_bnmode3'],
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:blue', ]
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

    for plot_y_value in ['success_rate', 'critic_q1']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle, max_seed=11, no_legend=no_legend, **d)

def plot_paper_fs_bn_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_bn_aggregate([env], no_legend=(env!='relocate'))

def plot_paper_fs_framestack_aggregate(envs=ADROIT_ALL_ENVS):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_framestack')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['3Frame', '2Frame', '1Frame']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs2_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:blue', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, **d)

def plot_paper_fs_pretanh_penalty_aggregate(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'fs', 'fs_pretanh_penalty')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['0', '0.001', '0.01', '0.1', '1']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.01_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.1_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_1_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:blue', 'tab:red', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate', 'abs_pretanh']:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)

def plot_paper_fs_pretanh_penalty_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_pretanh_penalty_aggregate([env], no_legend=(env!='relocate'))

def plot_extra_stage1_model(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'ex', 'ex_s1_model')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['resnet6-32channel', 'resnet6-64channel', 'resnet10-32channel', 'resnet10-64channel', 'resnet18-32channel', 'resnet18-64channel']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        [
            'vrl3_ar2_fs3_pixels_resnet6_64channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        [
            'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        [
            'vrl3_ar2_fs3_pixels_resnet10_64channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        [
            'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        [
            'vrl3_ar2_fs3_pixels_resnet18_64channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:blue', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate','success_rate_aug',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)
def plot_extra_stage1_model_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_extra_stage1_model([env])

def plot_paper_fs_framestack_per_task():
    for env in ADROIT_ALL_ENVS:
        plot_paper_fs_framestack_aggregate([env])


def plot_nrebuttal_CCE_ablation(envs=ADROIT_ALL_ENVS, no_legend=False, fs1=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'ex', 'cce_abl')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)
    # 'VRL3 64ch',
    labels = ['VRL3 32ch', 'Rand First', 'Encoding Sum', 'Encoding Concat', 'Encoding Diff Concat']
    if fs1:
        variants = [
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rflTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrue'],
            ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrueTrue']
        ]
    else:
        variants = [
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rflTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrue'],
            ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrueTrue']
        ]
    colors = ['tab:red', 'tab:blue', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, smooth=5, **d) # need better readability


def plot_nrebuttal_CCE_ablation_per_task():
    for env in ['door', 'hammer', 'pen', 'relocate']: # 'relocate' # 'door', 'hammer', 'pen',  #TODO add them after we finish new experiments
        fs1 = False
        if env == 'relocate':
            fs1 = True
        try:
            plot_nrebuttal_CCE_ablation([env], fs1=fs1)
        except Exception as e:
            print(e)

def plot_nrebuttal_CCE_ablation_fs1(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'ex', 'cce_abl_new')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)
    # 'VRL3 64ch',
    labels = ['VRL3 32ch', 'Rand First', 'Encoding Sum', 'Encoding Concat', 'Encoding Diff Concat']
    variants = [
        #['vrl3_ar2_fs3_pixels_resnet6_64channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rflTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrue'],
        ['vrl3_ar2_fs1_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrueTrue']
    ]
    colors = ['tab:red', 'tab:blue', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)


def plot_nrebuttal_CCE_ablation_with64(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'ex', 'cce_abl')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)
    # 'VRL3 64ch',
    labels = ['VRL3 32ch', 'Rand First', 'Encoding Sum', 'Encoding Concat', 'VRL3 64ch']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rflTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue_lfTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_64channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
    ]
    colors = ['tab:red', 'tab:blue', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)

def plot_nrebuttal_CCE_ablation2(envs=ADROIT_ALL_ENVS, no_legend=False):
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'ex', 'cce_abl2')
    list_of_data_dir = ['abl', 'ablph', 'ablneurips']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = ['VRL3', 'Rand first', 'Sum']
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rflTrue'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_cmmTrue'],
    ]
    colors = ['tab:red', 'tab:blue', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:orange', ]
    dashes = ['solid' for _ in range(6)]

    label2variants, label2colors, label2linestyle = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]
        label2linestyle[label] = dashes[i]

    for label, variants in label2variants.items():  # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    for plot_y_value in ['success_rate','success_rate_aug',]:
        save_parent_path = os.path.join(save_folder, plot_y_value)
        d = adroit_success_rate_default_min_max_dict if plot_y_value == 'success_rate' else adroit_other_value_default_min_max_dict
        save_name = '%s_%s' % (save_prefix, plot_y_value)
        print(save_parent_path, save_name)
        plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_parent_path, save_name=save_name,
                            plot_y_value=plot_y_value, label2colors=label2colors, label2linestyle=label2linestyle,
                            max_seed=11, no_legend=no_legend, **d)


def plot_paper_fs_s3_buffer():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_buffer':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb300000',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb30000',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb5000',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': '1M',
                         'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb300000':'300K',
                     'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb30000':'30K',
                     'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb5000':'5K',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb300000': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb30000': 'tab:olive',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_rb5000': 'tab:gray',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = -0.01
                    ymax = 1
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax, xmax=int(1e6) * NUM_MILLION_ON_X_AXIS,
                     no_legend=no_legend,
                     )

def plot_paper_compare_out_compress():
    list_of_data_dir = ['base_easy', 'fixed1', ]
    save_folder = os.path.join(base_save_dir, 'as') # as for additional studies
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    to_plot = [
        # 'drq_baseline_nstep3',
        # 'drq_nstep3_pt5_0.001',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nec_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128',

        'drq_nstep3_pt5_0.001_ftrue_numc2',
        # 'drq_nstep3_pt5_0.001_ftrue_numc16',
        'drq_nstep3_pt5_0.001_ftrue_numc32',
        'drq_nstep3_pt5_0.001_ftrue_numc64',
        # 'drq_nstep3_pt5_0.001_ftrue_numc96',
        'drq_nstep3_pt5_0.001_ftrue_numc128',
        # 'drq_nstep3_pt5_0.001_ftrue_numc192',
    ]

    variant2dashes = {
        'drq_nstep3_pt5_0.001_ftrue_numc2':True,
        'drq_nstep3_pt5_0.001_ftrue_numc32': True,
        'drq_nstep3_pt5_0.001_ftrue_numc64': True,
        'drq_nstep3_pt5_0.001_ftrue_numc128': True,

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': False,
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128': False,
    }

    variant2colors = {
        'drq_nstep3_pt5_0.001_ftrue_numc2': 'tab:gray',
        'drq_nstep3_pt5_0.001_ftrue_numc32': 'tab:blue',
        'drq_nstep3_pt5_0.001_ftrue_numc64': 'tab:orange',
        'drq_nstep3_pt5_0.001_ftrue_numc128': 'tab:red',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'tab:gray',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'tab:blue',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'tab:orange',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'tab:red',
    }

    variant2labels = {
        'drq_nstep3_pt5_0.001_ftrue_numc2': 'compress_out_numc2',
        'drq_nstep3_pt5_0.001_ftrue_numc32': 'compress_out_numc32',
        'drq_nstep3_pt5_0.001_ftrue_numc64': 'compress_out_numc64',
        'drq_nstep3_pt5_0.001_ftrue_numc128': 'compress_out_numc128',

        'drq_nec_nstep3_pt5_0.001_ftrue_numc2': 'conv_out_numc2',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc32':  'conv_out_numc32',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc64': 'conv_out_numc64',
        'drq_nec_nstep3_pt5_0.001_ftrue_numc128':  'conv_out_numc128',
    }
    envs = ["cartpole_swingup", "cheetah_run", "walker_walk",]
    for env in [
        # "cartpole_swingup",
        "cheetah_run", "walker_walk",
    ]:
        print(env)
        plot(paths, variants_to_plot=[
            v + '_' + env for v in to_plot
        ],
             plot_extra_y=True,
             variant2colors=expand_with_env_name(variant2colors, env),
             variant2dashes=expand_with_env_name(variant2dashes, env),
             variant2labels=expand_with_env_name(variant2labels, env)
             )

    save_prefix = 'freeze_out_compress'
    for env in envs:
        print(env, save_prefix)
        for y in [
            'performance',
        ]:
            save_parent_path = os.path.join(save_folder, y)
            save_name = '%s_%s_%s' % (save_prefix, env, y)
            y_to_plot = y if y != 'performance' else 'episode_reward'
            ymin, ymax = None, None
            if 'success_rate' in y_to_plot:
                ymin = DEFAULT_Y_MIN
                ymax = DEFAULT_Y_MAX
            no_legend = False
            plot(paths, variants_to_plot=[
                v + '_' + env for v in to_plot
            ],
                 plot_y_value=y_to_plot,
                 variant2colors=expand_with_env_name(variant2colors, env),
                 variant2labels=expand_with_env_name(variant2labels, env),
                 # variant2dashes=expand_with_env_name(variant2dashes, env),
                 smooth=10,
                 save_folder=save_parent_path,
                 save_name=save_name,
                 ymin=ymin, ymax=ymax, xmax=int(1e6) * 3,
                 no_legend=no_legend,
                 )

def plot_paper_hyper_sens(): # try plot ablations in the fancy way
    # base_path = os.path.join(base_dir, 'abl')
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'abl')

    to_abl = {
        'abl_hs_s1_pretrain':[
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                'pretrain',
                'random'
            ]
        ],
        'abl_hs_s2_bc_update': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc20000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc50000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '5000',
                '20000',
                '50000',
            ]
        ],
        'abl_hs_s2_cql_random': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr1_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr20_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr50_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr100_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '1',
                '10',
                '20',
                '50',
                '100',
            ]
        ],
        'abl_hs_s2_cql_std': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.01_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.5_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.01',
                '0.1',
                '0,5',
                '1',
            ]
        ],
        'abl_hs_s2_cql_update':[
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql0_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql5000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql100000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '5000',
                '30000',
                '100000',
            ]
        ],
        'abl_hs_s2_cql_weight': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w0.1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w10_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.1',
                '1',
                '10',
            ]
        ],
        'abl_hs_s2_enc_update': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                'update',
                'no update',
            ]
        ],
        'abl_hs_s3_bc_decay': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.9_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.95_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0.5',
                '0.9',
                '0.95',
            ]
        ],
        'abl_hs_s3_bc_weight': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.01_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.1_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc1_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.001',
                '0.01',
                '0.1',
                '1',
            ]
        ],
        'abl_hs_s3_std': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.2_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.5_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.01',
                '0.1',
                '0.2',
                '0.5',
            ]
        ],
        'abl_hs_s23_aug': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augFalse_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                'augment',
                'no augment',
            ]
        ],
        'abl_hs_s23_demo_bs': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs32_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs128_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs256_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '32',
                '64',
                '128',
                '256',
            ]
        ],
        'abl_hs_s23_enc_lr': [ #TODO for better consistency, we might want to make sure we are using the same data..
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.001',
                '0.01',
                '0.1',
                '1',
            ]
        ],
        'abl_hs_s23_lr': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr3e-05_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0003_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '3e-5',
                '1e-4',
                '3e-4',
                '1e-3',
            ]
        ],
        'abl_hs_s23_pretanh_weight': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.01_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.1_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_1_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                '0',
                '0.001',
                '0.01',
                '0.1',
                '1',
            ]
        ],
        'abl_hs_s23_safe_q_factor': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.3_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.9_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etTrue',
            ],
            [
                '0',
                '0.3',
                '0.5',
                '0.9',
                '1',
            ]
        ],
        'abl_hs_s23_safe_q_threshold': [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt50_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt1000_qf0.5_ef0_True_etTrue',
            ],
            [
                '50',
                '100',
                '200',
                '1000',
            ]
        ],
    }

    """
    # copy paste this for new ablations
        'abl_hs_s':[
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                'base',
            ]
        ],
    """

    envs = ['door', 'hammer', 'pen', 'relocate', ]
    #
    plot_one_figure_with_labels = True
    if plot_one_figure_with_labels:
        key = 'abl_hs_s1_pretrain'
        value = [
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
            [
                'pretrain',
                'random'
            ]
        ],
        save_name = key
        variants, categories = to_abl[key]
        plot_hyper_sensitivity(paths, variants, categories, save_folder=save_folder, save_name=save_name,
                               seed_max=2, envs=envs)  # for now just use 3 seeds for consistency
        quit()
    for key, value in to_abl.items():
        save_name = key
        variants, categories = to_abl[key]
        plot_hyper_sensitivity(paths, variants, categories,save_folder=save_folder,save_name=save_name,
                               seed_max=2, envs=envs) # for now just use 3 seeds for consistency

def plot_post_fs_pretanh():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s23_pretanh':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'BC5K,CQL30K',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'BC1M',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'BC1M-NP',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'CQL1M',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'CQL1M-NP',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:grey',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:grey',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
    }
    variant2dashes = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc1000000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': True,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': True,
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_fs_long_stage2(): # stage 2 with more CQL updates
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s2long':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                # 'rrl'
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'CQL30K',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'CQL1M',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
    }
    variant2dashes = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': False,
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_s1_model(): # stage 2 with more CQL updates
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s1_model_32ch':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_s1_model_32ch']
    labels = ['Resnet6','Resnet10','Resnet18','Resnet6_NP','Resnet10_NP','Resnet18_NP',]
    colors = ['tab:red', 'tab:orange', 'tab:blue'] * 2
    dashes = [False,False,False,True, True, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_model_elr():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        # "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_resnet10_32_elr':
            [
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
        'fs_resnet18_32_elr':
            [
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
        'fs_resnet6_32_elr':
        [
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
        ]
    }

    variants = prefix_to_list_of_variants['fs_resnet10_32_elr'] + prefix_to_list_of_variants['fs_resnet18_32_elr'] + prefix_to_list_of_variants['fs_resnet6_32_elr']
    labels = ['0','0.001','0.01','0.1','1',] + ['0','0.001','0.01','0.1','1',] + ['0','0.001','0.01','0.1','1',]
    colors = ['tab:red', 'tab:orange', 'tab:blue','tab:brown','tab:gray'] +  ['tab:red', 'tab:orange', 'tab:blue','tab:brown','tab:gray']+  ['tab:red', 'tab:orange', 'tab:blue','tab:brown','tab:gray']
    dashes = [False,False,False,False,False,] + [False,False,False,False,False,]+ [False,False,False,False,False,]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1',
                'abs_pretanh',
                'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_model_elr_p():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        # "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_resnet10_32_elr_p':
            [
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
        'fs_resnet18_32_elr_p':
            [
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_resnet10_32_elr_p'] + prefix_to_list_of_variants['fs_resnet18_32_elr_p']# + prefix_to_list_of_variants['fs_resnet6_32_elr']
    labels = ['0.001','0.01','0.1'] + ['0.001','0.01','0.1',] + ['0.001','0.01','0.1',] + ['0.001','0.01','0.1',] #+ ['0','0.001','0.01','0.1','1',]
    colors = ['tab:orange', 'tab:blue','tab:brown'] * 4
    dashes = [False,False,False, True, True, True] *2

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_model_elr_complete_results(): # some are missing in previous results, this one is the complete one
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_resnet6_32_elr_complete':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
        'fs_resnet10_32_elr_complete':
            [
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet10_32channel_pFalse_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
        'fs_resnet18_32_elr_complete':
            [
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet18_32channel_pFalse_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_resnet6_32_elr_complete'] + prefix_to_list_of_variants['fs_resnet10_32_elr_complete'] + prefix_to_list_of_variants['fs_resnet18_32_elr_complete']
    labels = ['1','0.1','0.01','0.001','0'] * 6
    colors = ['tab:red','tab:orange', 'tab:blue','tab:brown', 'tab:gray'] * 6
    dashes = [False,False,False,False,False, True, True, True, True, True] *3

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend, font_size=20, legend_size=20
                     )

def plot_post_framestack():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_framestack':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs2_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs1_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_framestack']
    labels = ['3Frame','2Frame','1Frame']
    colors = ['tab:red', 'tab:orange', 'tab:blue']
    dashes = [False,False,False]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_s2_qmin():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s2_qmin_cql30K':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin0_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-100_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-200_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin0_invTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-100_invTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-200_invTrue',
            ],
        'fs_s2_qmin_cql1M':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin0_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-100_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-200_invFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin0_invTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-100_invTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql1000000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_qmin-200_invTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_s2_qmin_cql30K'] + prefix_to_list_of_variants['fs_s2_qmin_cql1M']
    labels = ['VRL3','0','-100','-200','INV 0','INV -100','INV -200'] + ['VRL3','0','-100','-200','INV 0','INV -100','INV -200']
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:gray', 'tab:orange', 'tab:blue', 'tab:gray'] + ['tab:red', 'tab:orange', 'tab:blue', 'tab:gray', 'tab:orange', 'tab:blue', 'tab:gray']
    dashes = [False,False,False,False, True, True, True] + [False,False,False,False, True, True, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1',
                'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_samechannel():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_samechannel_s2con': # contrastive learning in s2
            [
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm0_s2con30000_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm1_s2con30000_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm2_s2con30000_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm3_s2con30000_s3neuFalse',
            ],
        'fs_samechannel_s2rl':
            [
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm0_s2con0_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm1_s2con0_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm2_s2con0_s3neuFalse',
                'vrl3_ar2_fs3_pixels_same_conv6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_sm3_s2con0_s3neuFalse',
            ],
    }

    variants = prefix_to_list_of_variants['fs_samechannel_s2con'] + prefix_to_list_of_variants['fs_samechannel_s2rl']
    labels = ['Con-SM0','Con-SM1','Con-SM2','Con-SM3',] + ['RL-SM0','RL-SM1','RL-SM2','RL-SM3',]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:gray'] +  ['tab:red', 'tab:orange', 'tab:blue', 'tab:gray']
    dashes = [False,False,False,False] + [False,False,False,False]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1',
                'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_mom_enc_elr():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_mom_enc_elr':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.001_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.01_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0.1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_mom_enc_elr']
    labels = ['No Momentum', '0', '0.001', '0.01', '0.1', '1']
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:gray']
    dashes = [False for _ in range(len(labels))]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_mom_enc_safeq():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_mom_enc_safeq':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt50_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt100_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_momTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt1000_qf0.5_ef0_True_etTrue_momTrue',
            ],
    }

    variants = prefix_to_list_of_variants['fs_mom_enc_safeq']
    labels = ['No Momentum', '50', '100', '200', '1000']
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:brown']
    dashes = [False for _ in range(len(labels))]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )


def plot_post_fs_s3_freeze_extra():
    list_of_data_dir = ['abl']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'fs')
    envs = [
        "door",
        # "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s3_freeze_extra':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etTrue_s3feTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etTrue_s3feTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef3_True_etTrue_s3feTrue',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':FRAMEWORK_NAME,
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'S1',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etTrue_s3feTrue': 'S1-EF1',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etTrue_s3feTrue':'S1-EF2',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef3_True_etTrue_s3feTrue':'S1-EF3',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es0_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue':'tab:orange',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef1_True_etTrue_s3feTrue':'tab:blue',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef2_True_etTrue_s3feTrue':'tab:purple',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eFalse_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef3_True_etTrue_s3feTrue':'tab:gray',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                'critic_loss', 'critic_q1'
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_dmc_medium():
    list_of_data_dir = ['dmc', 'dmc_newhyper_3seed', 'drqv2_author']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc')
    envs = [
        "acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy", "finger_turn_hard",
        "hopper_hop", "quadruped_run", "quadruped_walk", "reach_duplo", "reacher_easy", "reacher_hard", "walker_run",
    ]

    prefix_to_list_of_variants = {
        'dmc_medium':
            [
                'drqv2_author',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    variants = prefix_to_list_of_variants['dmc_medium']
    labels = ['def_s3', 'R6_s3', 'R6_s13', 'R6_s23', 'R6_s123',]
    colors = ['tab:blue', 'tab:red','tab:orange', 'tab:purple','tab:gray', ]
    dashes = [False, False, False, True, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            if env in DMC_HARD_ENVS:
                x_max = DEFAULT_DMC_HARD_X_MAX
            else:
                x_max = DEFAULT_DMC_X_MAX
            for y in [
                'performance',
                # 'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=x_max,
                     no_legend=no_legend,
                     )

def plot_post_dmc_easy():
    list_of_data_dir = ['dmc', 'dmc_newhyper_3seed', 'drqv2_author']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc')
    envs = [
        "cartpole_balance", "cartpole_balance_sparse", "cartpole_swingup", "cup_catch", "finger_spin", "hopper_stand", "pendulum_swingup", "walker_stand", "walker_walk"
    ]

    prefix_to_list_of_variants = {
        'dmc_easy':
            [
                'drqv2_author',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    variants = prefix_to_list_of_variants['dmc_easy']
    labels = ['def_s3', 'R6_s3', 'R6_s13', 'R6_s23', 'R6_s123',]
    colors = ['tab:blue', 'tab:red','tab:orange', 'tab:purple','tab:gray', ]
    dashes = [False, False, False, True, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'performance',
                # 'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_DMC_EASY_X_MAX,
                     no_legend=no_legend,
                     )

def plot_post_dmc_hard():
    list_of_data_dir = ['dmc', 'dmc_newhyper_3seed', 'drqv2_author']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc')
    envs = [
        "humanoid_stand", "humanoid_walk", "humanoid_run"
    ]

    prefix_to_list_of_variants = {
        'dmc_hard':
            [
                'drqv2_author',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    variants = prefix_to_list_of_variants['dmc_hard']
    labels = ['def_s3', 'R6_s3', 'R6_s13', 'R6_s23', 'R6_s123',]
    colors = ['tab:blue', 'tab:red','tab:orange', 'tab:purple','tab:gray', ]
    dashes = [False, False, False, True, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'performance',
                # 'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_DMC_HARD_X_MAX,
                     no_legend=no_legend,
                     max_frames=DEFAULT_DMC_HARD_X_MAX
                     )

def plot_post_dmc_medium_aggregate():
    list_of_data_dir = ['dmc']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc-aggregate')
    envs = [
        "acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy", "finger_turn_hard",
        "hopper_hop", "quadruped_run", "quadruped_walk", "reach_duplo", "reacher_easy", "reacher_hard", "walker_run"
    ]

    label2variants = {
        'default':['vrl3_ar2_fs3_pixels_none_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse'],
        'resnet6': [
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse'],
        'resnet6_p': [
            'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse'],
    }

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    save_prefix = 'vrl3-dmc-medium-aggregate'
    y_to_plot = 'performance'
    save_name = '%s_%s' % (save_prefix, y_to_plot)
    plot_aggregate(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name)

def check_dmc_vrl3_medium():
    dir = os.path.join(base_dir, 'dmc_hyper')
    # dir = os.path.join(base_dir, 'adroit-relocatehyper') # this is with rrl features
    task = 'walker_run'
    eval_criteria = 'episode_reward'
    hyper_search_dmc([dir], task_name=task, start_frame=500000, end_frame=1500000, eval_criteria=eval_criteria)


def check_dmc_vrl3_hard():
    dir = os.path.join(base_dir, 'dmc_hard_hyper1')
    # dir = os.path.join(base_dir, 'adroit-relocatehyper') # this is with rrl features
    task = 'humanoid_run'
    eval_criteria = 'episode_reward'
    hyper_search_dmc([dir], task_name=task, start_frame=15000000, end_frame=17000000, eval_criteria=eval_criteria)


def plot_post_dmc_medium_hyper2(): # just for "walker_run", "cheetah_run", "quadruped_run"
    list_of_data_dir = ['dmc', 'dmc_hyper2']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc')
    envs = [
        "walker_run", "cheetah_run", "quadruped_run"
    ]

    prefix_to_list_of_variants = {
        'dmc_medium_h2':
            [
                'vrl3_ar2_fs3_pixels_none_pFalse_augTrue_es1_dbs64_pt5_0_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.01_bc0_0.5_s2d0_sa4000_ra2000_demo0_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3stdauto_bc0_0.5_s2d0_sa0_ra0_demo25_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3stdauto_bc0_0.5_s2d0_sa0_ra0_demo25_qt1000_qf0.5_ef0_True_etFalse',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse',
            ],
    }

    variants = prefix_to_list_of_variants['dmc_medium_h2']
    labels = ['def_s3', 'R6_s23', 'R6_s123', 'R6_s123_pt.001_std.1']
    colors = ['tab:blue', 'tab:red','tab:orange', 'tab:purple' ]
    dashes = [False, False, False, True]

    variant2labels, variant2colors, variant2dashes = {}, {}, {}
    for i, v in enumerate(variants):
        variant2labels[v] = labels[i]
        variant2colors[v] = colors[i]
        variant2dashes[v] = dashes[i]

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            if env in DMC_HARD_ENVS:
                x_max = DEFAULT_DMC_HARD_X_MAX
            else:
                x_max = DEFAULT_DMC_X_MAX
            for y in [
                'performance',
                # 'success_rate',
                # 'critic_loss', 'critic_q1',
                # 'abs_pretanh',
                # 'max_abs_pretanh',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=x_max,
                     no_legend=no_legend,
                     )

def plot_post_dmc_rebuttal_aggregate():
    list_of_data_dir = ['dmc', 'dmc_newhyper_3seed', 'drqv2_author']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'dmc-aggregate')
    envs = [
        "acrobot_swingup", "cartpole_swingup_sparse", "cheetah_run", "finger_turn_easy", "finger_turn_hard",
        "hopper_hop", "quadruped_run", "quadruped_walk", "reach_duplo", "reacher_easy", "reacher_hard", "walker_run"
    ]

    label2variants = {
        # 'DrQv2':['drqv2_author'],
        'VRL3(S123)': ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse'],
    }

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    save_prefix = 'vrl3-dmc-rebuttal-aggregate'
    y_to_plot = 'performance'
    save_name = '%s_%s' % (save_prefix, y_to_plot)
    plot_aggregate(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name)

def plot_dmc_aggregate_all24(envs = DMC_ALL_ENVS, xmax=int(1e6)+50000, drqv2_max_frame=int(1e6)-1000, max_frames=int(1e6), no_legend=False): # for neurips
    print(envs)
    list_of_data_dir = ['dmc', 'dmc_newhyper_3seed', 'drqv2_author']
    save_folder_name, save_prefix = decide_placeholder_prefix(envs, 'dmc', 'dmc_main')
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, save_folder_name)

    labels = [
        'VRL3(S123)',
        'DrQv2',
        'DrQv2fD',
        'FERM(VRL3)',
    ]
    variants = [
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etFalse'],
        ['drqv2_author'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etFalse'],
        ['vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_es1_dbs64_pt5_0.001_s2eTrue_bc0_cql0_w1_s2std0.1_nr10_s3std0.1_bc0_0.5_s2d0_sa0_ra0_demo25_qt200_qf1_ef0_True_etFalse_s2contrast30000']
    ]
    colors = ['tab:red', 'tab:cyan', 'tab:blue', 'tab:orange']
    label2variants, label2colors, label2dashes = {}, {}, {}
    for i, label in enumerate(labels):
        label2variants[label] = variants[i]
        label2colors[label] = colors[i]

    for label, variants in label2variants.items(): # add environment
        variants_with_envs = []
        for variant in variants:
            for env in envs:
                variants_with_envs.append(variant + '_' + env)
        label2variants[label] = variants_with_envs

    plot_y_value = 'episode_reward'
    save_name = '%s_%s' % (save_prefix, plot_y_value)
    save_folder = os.path.join(save_folder, plot_y_value)
    print(save_folder, save_name)
    plot_aggregate_0506(paths, label2variants=label2variants, save_folder=save_folder, save_name=save_name,
                        plot_y_value=plot_y_value, label2colors=label2colors, xmin=DEFAULT_X_MIN, xmax=xmax, n_boot=20, smooth=10, max_seed=11,
                        drqv2_max_frame=drqv2_max_frame, max_frames=max_frames, no_legend=no_legend)

def plot_dmc_main_per_task():
    for env in DMC_EASY_ENVS:
        no_legend = False if env == 'walker_walk' else True
        plot_dmc_aggregate_all24(envs=[env], no_legend=no_legend)
    for env in DMC_MEDIUM_ENVS:
        no_legend = False if env == 'walker_run' else True
        plot_dmc_aggregate_all24(envs=[env], xmax=int(3e6)+50000, drqv2_max_frame=int(3e6), max_frames=int(3e6), no_legend=no_legend)
    for env in DMC_HARD_ENVS:
        no_legend = False if env == 'humanoid_run' else True
        plot_dmc_aggregate_all24(envs=[env], xmax=int(3e7)+50000, drqv2_max_frame=int(3e7), max_frames=int(3e7), no_legend=no_legend)

def plot_may_pen_hammer_test():
    list_of_data_dir = ['abl', 'ablph']
    paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
    save_folder = os.path.join(base_save_dir, 'trial')
    envs = [
        "door",
        "pen", "hammer",
        "relocate"
    ]

    prefix_to_list_of_variants = {
        'fs_s123_encoder_finetune':
            [
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
                'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue',
            ],
    }
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc0_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen
    # vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue_pen_s0
    variant2labels = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S123',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'S23',
    }
    variant2colors = {
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pTrue_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:red',
        'vrl3_ar2_fs3_pixels_resnet6_32channel_pFalse_augTrue_lr0.0001_esauto_dbs64_pt5_0.001_s2eTrue_bc5000_cql30000_w1_s2std0.1_nr10_s3std0.01_bc0.001_0.5_s2d0_sa0_ra0_demo25_qt200_qf0.5_ef0_True_etTrue': 'tab:orange',
    }

    plot_this_time = prefix_to_list_of_variants.keys()
    for save_prefix in plot_this_time:
        to_plot = prefix_to_list_of_variants[save_prefix]
        for env in envs:
            print(env, save_prefix)
            for y in [
                'success_rate',
                      ]:
                save_parent_path = os.path.join(save_folder, y)
                save_name = '%s_%s_%s' % (save_prefix, env, y)
                y_to_plot = y if y != 'performance' else 'episode_reward'
                ymin, ymax = None, None
                if 'success_rate' in y_to_plot:
                    ymin = DEFAULT_Y_MIN
                    ymax = DEFAULT_Y_MAX
                no_legend = False if env == 'relocate' else True
                plot(paths, variants_to_plot=[
                    v + '_' + env for v in to_plot
                ],
                     plot_y_value=y_to_plot,
                     variant2colors=expand_with_env_name(variant2colors, env),
                     variant2labels=expand_with_env_name(variant2labels, env),
                     # variant2dashes=expand_with_env_name(variant2dashes, env),
                     smooth=10,
                     save_folder=save_parent_path,
                     save_name=save_name,
                     ymin=ymin, ymax=ymax,
                     xmin=DEFAULT_X_MIN, xmax=DEFAULT_X_MAX,
                     no_legend=no_legend,
                     )



# plot_compare_out_compress()
# plot_compare_out_compress2()

# plot_resnet_new_cap()
# plot_resnet_new_numc()

# plot_resnet_compare_to_old_structure()

# plot_resnet_end_end()
# plot_fiencoder()


# plot_resbase()
# plot_atc()
# plot_atc_dense()
# plot_distill_change_weight_and_distill_index()

# plot_distill_low_cap()
# plot_distill_change_ul_cap()

# plot_distill_change_ul_cap()

# plot_compare_distill_to_baseline()

# plot_compare_n_compress()

"""
put fully ready data plotting functions below (paper-ready plots)
"""

# plot_double()
# plot_intermediate_standard_conv()
# plot_intermediate_standard_conv_freeze()
# plot_intermediate_resnet()
# plot_intermediate_resnet_freeze()



# plot_double_res_additional()
# plot_double_res_additional_with_baseline()
# plot_single_res_policy_loss_gradient()
# plot_double_standard_policy_loss_gradient()
# plot_naivelarge()

# plot_medium12_3seed_doubleres_baseline_compare()
# plot_medium6task_3seed_baseline_variants()
# plot_pt_rrl()

# plot_utd()
# plot_stage3_conv2()
# plot_stage3_conv2_debug()
# plot_stage2_conv4()
# plot_double_res_human()
# plot_ssl_1()
# plot_ssl_1_aggregate()
# plot_ssl_stage2()
# plot_ssl_stage2_extra()
# plot_ssl_stage2_aggregate()

# plot_atest()
# plot_atest2() # for none, ssl, tl, plot a figure for each env on success rate...
# plot_atest2_aggregate()
# plot_atest2_others()
# plot_atest2_success_rate_best()
# plot_atest2_no_data_aug()
# plot_atest2_small_lr()
# plot_atest3()
# check_adroit_first_group() # hypersearch
# plot_atest2_small_lr()
# plot_atest2_small_lr_old()
# plot_drq_with_rrl_resnet_feature()
# plot_adroit_stable()
# plot_adroit_stable_distraction_success_rate()
# check_adroit_stable_group()
# check_adroit_stable_group()
# check_adroit_stable_group()
# check_adroit_stable_group()
# plot_atest2_stable_hyper_door()
# plot_atest2_stable_hyper_hammer()
# check_adroit_stable_group2()
# plot_atest_stable_small_policy_lr()
# check_adroit_cql()
# plot_1231_hyper3()
# plot_1231_relocate()
# check_relocate()
# check_res6pen()
# plot_1231_hyper6_stage2_up_encoder_only()
# plot_10seed_relocate()
# check_test()
# plot_relocate_first_ablation()
# plot_relocate_second_ablation()
# plot_abs2_more()
# plot_main()
# plot_main_relocate()
# plot_main2()
# plot_main2_analysis()
# plot_varl3_main_door_hammer_pen()
# plot_varl3_main_relocate()
# plot_main4()


# paper: run this to generate hyper sensitivity plots #TODO might want to set this part to use 3 seeds...
# plot_paper_hyper_sens()

# paper: main comparison with RRL
# plot_paper_main()

# plot_paper_fs_s23_q_safe_factor()

## following 3: use first 2, or the 3rd
# plot_paper_fs_s1_pretrain()
# plot_paper_fs_s23_encoder_finetune()
# plot_may_pen_hammer_test()

###############################################################################
############################################################################### 05092022 working on this now
def plot_neurips():
    ## figure 2, main result on Adroit and DMC
    plot_paper_main_more_aggregate()
    plot_dmc_aggregate_all24()

    ## figure 3, BC/disable, S2-S3 sucess and Q value, encoder lr
    # plot_paper_fs_s2_bc()
    plot_paper_fs_s2_bc_aggregate()

    # plot_paper_fs_s23_transition()
    plot_paper_fs_s23_transition_q_safe_factor_aggregate()

    # plot_paper_fs_s23_enc_lr_scale()
    plot_paper_fs_s23_enc_lr_scale_aggregate()

    ## figure 4
    # plot_paper_fs_s123_encoder_update()
    plot_paper_fs_s123_stage_update_aggregate()
    plot_paper_fs_s123_encoder_update_aggregate()

    ## figure 5 hyper sensitivity
def plot_neurips_appendix():
    plot_paper_main_more_per_task()
    plot_paper_fs_s2_bc_per_task()
    plot_paper_fs_s23_transition_q_safe_factor_per_task()
    plot_paper_fs_s23_enc_lr_scale_per_task()
    plot_paper_fs_s123_stage_update_per_task()
    plot_paper_fs_s123_encoder_update_per_task()

    plot_paper_fs_bn_per_task()
    # plot_paper_fs_framestack_per_task()
    plot_paper_fs_pretanh_penalty_per_task()

# plot_paper_fs_s123_encoder_update_aggregate()
# plot_paper_fs_s23_transition_q_safe_factor_aggregate()
# quit()
# plot_paper_fs_s23_transition_q_safe_factor_aggregate()
# plot_dmc_aggregate_all21()

# plot_dmc_aggregate_all24()
# plot_dmc_main_per_task()
# plot_dmc_aggregate_all24(envs=['walker_run'], xmax=int(3e6)+50000, drqv2_max_frame=int(3e6), max_frames=int(3e6), no_legend=True)

# plot_paper_fs_s2_bc_aggregate()
# plot_paper_fs_s2_bc_per_task()
# plot_paper_fs_s23_transition_q_safe_factor_aggregate()
# plot_paper_fs_s23_transition_q_safe_factor_per_task()


# plot_paper_fs_s23_enc_lr_scale_aggregate()
# plot_paper_fs_s23_enc_lr_scale_per_task()

# plot_paper_fs_s123_stage_update_aggregate()
# plot_paper_fs_s123_stage_update_per_task()


# plot_paper_fs_s123_encoder_update_aggregate()
# plot_paper_fs_s123_encoder_update_per_task()

# plot_paper_fs_bn_per_task()
# plot_paper_fs_bn_aggregate()
# plot_paper_fs_framestack_aggregate()
# plot_paper_fs_framestack_per_task()

# plot_paper_fs_s2_extrabc_aggregate()
# plot_paper_fs_pretanh_penalty_aggregate()
# plot_paper_fs_pretanh_penalty_per_task()
# plot_neurips_appendix()
# plot_paper_fs_pretanh_penalty_per_task()
# plot_extra_stage1_model()
# plot_extra_stage1_model_per_task()

# plot_paper_fs_s123_encoder_update_aggregate_extra()
# plot_paper_fs_s123_encoder_per_task_extra()

# plot_paper_fs_s23_enc_lr_scale_aggregate()

# plot_extra_stage1_model_per_task()

# door_hammer_env = ['door', 'hammer']

# plot_nrebuttal_CCE_ablation(envs=door_hammer_env)

# plot_nrebuttal_CCE_ablation_per_task()
# plot_nrebuttal_long_term_per_task()

# for env in ['door', 'hammer']:
#     plot_nrebuttal_CCE_ablation2(envs=[env,])

# plot_nrebuttal_long_term()

# plot_nrebuttal_CCE_ablation_per_task()

# plot_nrebuttal_CCE_ablation(['door', 'hammer', 'pen'])

# plot_nrebuttal_CCE_ablation_per_task()
# plot_nrebuttal_CCE_ablation()

# plot_paper_fs_framestack_per_task()

# TODO need to replot CCE ablation for other 3 envs...
# plot_nrebuttal_CCE_ablation_fs1(envs=['relocate',])

plot_nrebuttal_s1_ssl(envs=['relocate',])


# =======================================================================================
# TODO TODO working here now
# plot_nrebuttal_CCE_ablation_per_task() # newest cce results, with bugs resolved
# plot_nrebuttal_bc_new_abl_per_task()
# plot_nrebuttal_bc_new_abl_nos1_per_task()

plot_nrebuttal_highcap_10(envs=['door',])
plot_nrebuttal_highcap_10(envs=['relocate',])
plot_nrebuttal_highcap_18(envs=['door',])
plot_nrebuttal_highcap_18(envs=['relocate',])

quit()


plot_paper_main_more_per_task()
quit()

plot_paper_hyper_sens()
quit()

plot_paper_main_relocate()
quit()
plot_neurips()
quit()
plot_paper_hyper_sens()

quit()

quit()

# plot_paper_fs_s3_buffer()

# def plot_in_paper_order():
#     plot_paper_fs_s123_encoder_update()
#     plot_paper_fs_s2_bc()
#     plot_paper_fs_s23_q_safe_factor()
#
#     plot_paper_fs_s23_transition()
#
#     plot_paper_fs_s23_enc_lr_scale()
#     plot_paper_fs_bn()
#     plot_paper_main_more()
    # plot_paper_compare_out_compress()



# plot_in_paper_order()
# plot_paper_fs_s2_bc()
# plot_paper_main_more()

#TODO add margin and make lines thicker and redo the plot...

def plot_post_submission():
    # plot_post_fs_pretanh()
    # plot_paper_fs_bn()
    # plot_post_fs_s3_freeze_extra()
    # plot_post_fs_long_stage2()
    # plot_post_s1_model()
    # plot_post_framestack()
    # plot_post_mom_enc_elr()
    # plot_post_mom_enc_safeq()
    # plot_post_model_elr()
    # plot_post_s2_qmin()
    # plot_post_samechannel()
    # plot_post_model_elr_p()
    # plot_post_model_elr_complete_results()
    pass
# plot_post_submission()

# plot_post_dmc_hard()
# plot_post_dmc_easy()
# plot_post_dmc_hard()
# plot_post_dmc_medium_aggregate()
# check_dmc_vrl3_medium()
# plot_post_dmc_medium_hyper2()
# check_dmc_vrl3_hard()
# plot_post_dmc_medium()
# plot_post_dmc_easy()
# plot_post_dmc_hard()
# plot_post_dmc_rebuttal_aggregate()





