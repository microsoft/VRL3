# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: plotting helper code is currently being cleaned up

import os

import numpy as np
import pandas
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# TODO add a flag for save all figures
from distutils.dir_util import copy_tree
from pathlib import Path

from plot_config import DEFAULT_BASE_DATA_PATH, DEFAULT_AMLT_PATH, DEFAULT_BASE_SAVE_PATH
np.set_printoptions(precision=4)

def move_data_from_amlt(src, dst):
    # given the amlt experiment name, when called, will move data from that folder into the
    # folder we desire...
    # assume the given experiment folder is the one that we download with amlt (e.g. a folder called valued-sparrow)
    # assume the move-to folder is the one in the data folder
    # this function will look at the exp folder, iterate through all folders there (call it subfolder), and for each of these subfolders
    # will look at its content, and move that to the data folder
    source = os.path.join(DEFAULT_AMLT_PATH, src)
    dest = os.path.join(DEFAULT_BASE_DATA_PATH, dst)
    print('move from:', source, 'to:', dest)

    if not os.path.isdir(dest):
        path = Path(dest)
        path.mkdir(parents=True)

    for topfolder_name in os.listdir(source):
        subfolder_path = os.path.join(source, topfolder_name)
        try:
            copy_tree(subfolder_path, dest, update=1)
        except Exception as e:
            print('move_data_from_amlt function exception:',e)


def hyper_search(paths, eval_criteria='episode_reward', task_name='', before_step=300):
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    for p in paths:
        print("search for path:", p)
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                name_no_seed = folder_name[:-3]

                if task_name in folder_name:
                    try:
                        train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except:
                        print(folder_name, "empty")

    # now we have variant name -> seeds dictionary, we can compute things...
    for variant in variant2seeds_eval.keys():
        seeds = variant2seeds_eval[variant]
        scores = []
        for seed in seeds:
            score = seed[eval_criteria][:before_step]
            scores.append(score.mean())
        score_ave_over_seeds = np.mean(scores)
        score_std_over_seeds = np.std(scores)
        variant2score[variant] = score_ave_over_seeds
        variant2scorestd[variant] = score_std_over_seeds

    # e.g. [('drq_baseline_nstep3_hopper_hop', 159.6003039010942), ('drq_nstep3_pt5_0.001_hopper_hop', 202.18650410831432)]
    variant_score_sorted = sorted(variant2score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for item in variant_score_sorted:
        variant = item[0]
        score_mean = item[1]
        score_std = variant2scorestd[variant]

        print('%s\t%.2f +- %.2f' % (variant, score_mean, score_std))

# TODO to do a hyperparameter evaluation for adroit...
#  we should try to
def hyper_search_adroit(paths, eval_criteria='episode_reward', task_name='', start_frame=0, end_frame=np.inf):
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    # need to specify start and end frame
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    for p in paths:
        print("search for path:", p)
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                name_no_seed = folder_name[:-3]

                if task_name in folder_name:
                    try:
                        train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except:
                        print(folder_name, "empty")

    # now we have variant name -> seeds dictionary, we can compute things...

    variant2q = {}
    variant2qloss = {}
    variant2pretanh = {}
    variant2maxpretanh = {}
    variant2infodict = {} # use variant name to refer to a dictionary that contains log info...
    for variant in variant2seeds_train.keys():
        seeds = variant2seeds_train[variant]
        scores = []
        losses = []
        pretanh = []
        max_pretanh = []
        variant2infodict[variant] = {}
        for seed in seeds:
            frame_array = seed['frame']
            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
            eval_values = seed['critic_q1'][idxs]
            if eval_values.shape[0] > 0:
                scores.append(eval_values)
                losses = seed['critic_loss'][idxs]
                pretanh = seed['abs_pretanh'][idxs]
                max_pretanh = seed['max_abs_pretanh'][idxs]

        if len(scores)>0:
            scores = np.concatenate(scores, 0)
            score_ave_over_seeds = np.mean(scores)
            score_std_over_seeds = np.std(scores)
            variant2q[variant] = score_ave_over_seeds
            variant2qloss[variant] = np.mean(losses)
            variant2pretanh[variant] = np.mean(pretanh)
            variant2maxpretanh[variant] = np.mean(max_pretanh)

    for variant in variant2seeds_eval.keys():
        seeds = variant2seeds_eval[variant]
        scores = []
        for seed in seeds:
            frame_array = seed['frame']
            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
            try:
                eval_values = seed[eval_criteria][idxs]
                if eval_values.shape[0] > 0:
                    scores.append(eval_values)
            except:
                print('not valid:',variant)

        if len(scores)>0:
            scores = np.concatenate(scores, 0)
            score_ave_over_seeds = np.mean(scores)
            score_std_over_seeds = np.std(scores)
            variant2score[variant] = score_ave_over_seeds
            variant2scorestd[variant] = score_std_over_seeds

    print('Evaluation criteria: %s, start frame %d, end frame %d' % (eval_criteria, start_frame, end_frame))
    # e.g. [('drq_baseline_nstep3_hopper_hop', 159.6003039010942), ('drq_nstep3_pt5_0.001_hopper_hop', 202.18650410831432)]
    variant_score_sorted = sorted(variant2score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for item in variant_score_sorted:
        variant = item[0]
        score_mean = item[1]
        score_std = variant2scorestd[variant]
        q = variant2q[variant]
        l = variant2qloss[variant]
        p = variant2pretanh[variant]
        maxp = variant2maxpretanh[variant]

        print('%s\t%.2f +- %.2f\t %.1f \t %.2f \t %.2f \t %.2f' % (variant, score_mean, score_std, q, l, p, maxp))

def hyper_search_dmc(paths, eval_criteria='episode_reward', task_name='', start_frame=0, end_frame=np.inf):
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    # need to specify start and end frame
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    for p in paths:
        print("search for path:", p)
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                name_no_seed = folder_name[:-3]

                if task_name in folder_name:
                    try:
                        train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except:
                        print(folder_name, "empty")

    # now we have variant name -> seeds dictionary, we can compute things...

    variant2q = {}
    variant2qloss = {}
    variant2pretanh = {}
    variant2maxpretanh = {}
    variant2infodict = {} # use variant name to refer to a dictionary that contains log info...
    for variant in variant2seeds_train.keys():
        seeds = variant2seeds_train[variant]
        scores = []
        losses = []
        pretanh = []
        max_pretanh = []
        variant2infodict[variant] = {}
        for seed in seeds:
            frame_array = seed['frame']
            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
            eval_values = seed['critic_q1'][idxs]
            if eval_values.shape[0] > 0:
                scores.append(eval_values)
                losses = seed['critic_loss'][idxs]
                pretanh = seed['abs_pretanh'][idxs]
                max_pretanh = seed['max_abs_pretanh'][idxs]

        if len(scores)>0:
            scores = np.concatenate(scores, 0)
            score_ave_over_seeds = np.mean(scores)
            score_std_over_seeds = np.std(scores)
            variant2q[variant] = score_ave_over_seeds
            variant2qloss[variant] = np.mean(losses)
            variant2pretanh[variant] = np.mean(pretanh)
            variant2maxpretanh[variant] = np.mean(max_pretanh)

    for variant in variant2seeds_eval.keys():
        seeds = variant2seeds_eval[variant]
        scores = []
        for seed in seeds:
            frame_array = seed['frame']
            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
            try:
                eval_values = seed[eval_criteria][idxs]
                if eval_values.shape[0] > 0:
                    scores.append(eval_values)
            except Exception as e:
                print(e)
                print('not valid:',variant)

        if len(scores)>0:
            scores = np.concatenate(scores, 0)
            score_ave_over_seeds = np.mean(scores)
            score_std_over_seeds = np.std(scores)
            variant2score[variant] = score_ave_over_seeds
            variant2scorestd[variant] = score_std_over_seeds

    print('Evaluation criteria: %s, start frame %d, end frame %d' % (eval_criteria, start_frame, end_frame))
    # e.g. [('drq_baseline_nstep3_hopper_hop', 159.6003039010942), ('drq_nstep3_pt5_0.001_hopper_hop', 202.18650410831432)]
    variant_score_sorted = sorted(variant2score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for item in variant_score_sorted:
        variant = item[0]
        score_mean = item[1]
        score_std = variant2scorestd[variant]
        q = variant2q[variant]
        l = variant2qloss[variant]
        p = variant2pretanh[variant]
        maxp = variant2maxpretanh[variant]

        print('%s\t%.2f +- %.2f\t %.1f \t %.2f \t %.2f \t %.2f' % (variant, score_mean, score_std, q, l, p, maxp))

def do_smooth(x, smooth):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x

def combine_data_in_seeds(seeds, column_name, skip, smooth=1):
    vals_list = []
    for d in seeds:
        vals_to_use = d[column_name].reshape(-1)

        # yhat = savgol_filter(vals_to_use, 21, 3)  # window size 51, polynomial order 3
        # TODO might want to use sth else...
        if smooth > 1 and smooth <= len(vals_to_use):
            yhat = do_smooth(vals_to_use, smooth)
        else:
            yhat = vals_to_use

        if skip > 1:
            yhat = yhat[::skip]
        vals_list.append(yhat)
    return np.concatenate(vals_list)

# TODO we can try allow aggregate plotting here...
def plot(paths, variants_to_plot=[], plot_y_value='episode_reward', plot_extra_y=False, extra_train_y_values=[], fig_size=[7, 7],
         variant2dashes=None, variant2colors=None, variant2labels=None, smooth=1, save_folder=None, save_name='', data_skip=1, y_log_scale=False, xmin=0, xmax=4050000,
         ymin=None, ymax=None, max_frames=4000000, no_legend=False, linewidth=7, font_size=28, legend_size=28, y_label=None, x_label='Frames',
         max_seed=3):
    sns.set(rc={'figure.figsize': fig_size})
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    if plot_y_value == 'episode_reward' or plot_y_value == 'success_rate' or plot_y_value == 'success_rate_aug':
        no_train_data = True
    else:
        no_train_data = False
    if y_label is None:
        y_label = decide_y_label(plot_y_value)
    for p in paths:
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                seed_string_index = folder_name.rfind('s')
                name_no_seed = folder_name[:seed_string_index-1]
                seed_value = int(folder_name[seed_string_index+1:])
                if seed_value >= max_seed:
                    continue

                if name_no_seed in variants_to_plot:
                    print(name_no_seed)
                    try:
                        if not no_train_data:
                            train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        # TODO need to generalize this
                        start_frame = 0
                        end_frame = max_frames
                        frame_array = eval_data['frame']
                        idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
                        eval_data = eval_data[idxs]

                        if not no_train_data:
                            frame_array = train_data['frame']
                            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
                            train_data = train_data[idxs]

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        if not no_train_data:
                            variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except Exception as e:
                        print(name_no_seed, 'empty', e)

    # plot here
    for variant in variants_to_plot:
        color = variant2colors[variant] if variant2colors is not None else None
        dash = variant2dashes[variant] if variant2dashes is not None else None
        label = variant2labels[variant] if variant2labels is not None else variant
        if dash:
            linestyle = 'dashed'
        else:
            linestyle = None

        if plot_y_value=='episode_reward' or plot_y_value=='success_rate' or plot_y_value=='success_rate_aug':
            seeds = variant2seeds_eval[variant]
        else:
            seeds = variant2seeds_train[variant]

        """this is to compute how many frames it took for us to reach e.g. 90% success rate"""
        check_success = False
        success_rate_threshold = 0.5
        if check_success:
            print("====================")
            print(variant)
            success_rate_seeds = []
            min_len = 999999999
            for i in range(len(seeds)):
                success_rate_seeds.append(seeds[i]['success_rate'].reshape(1,-1))
                seed_len = seeds[i]['success_rate'].shape[0]
                if min_len > seed_len:
                    min_len = seed_len
            print("min seed len:",min_len)
            for i in range(len(seeds)):
                success_rate_seeds[i] = success_rate_seeds[i][:,:min_len]
            average_success = np.concatenate(success_rate_seeds, axis=0) # 3x82
            # print(average_success.shape)
            # print(average_success)
            # quit()
            average_success = average_success.mean(axis=0)

            # for any 200000 frames, if one can get an average of >= 90% performance, then at the end of 200000 frames, that is where we say it first achieved 90% success
            if 'rrl' in variant:
                frames_per_iter = 40000
            else:
                frames_per_iter = 50000
            iter_window = int(200000/frames_per_iter)
            print("windows", iter_window, 'seeds:', len(seeds))

            number_frames = 12000000
            for i in range(len(average_success) - iter_window):
                current_score = average_success[i:i+iter_window].mean()
                if current_score >= success_rate_threshold:
                    number_frames = (i + iter_window) * frames_per_iter
                    break
            #
            # first_achieved = np.argmax(average_success > 0.9)
            # if 'rrl' in variant:
            #     number_frames = first_achieved * 40000
            # else:
            #     number_frames = first_achieved * 50000
            print(number_frames)
            print("------")

        x = combine_data_in_seeds(seeds, 'frame', skip=data_skip)
        y = combine_data_in_seeds(seeds, plot_y_value, skip=data_skip, smooth=smooth)
        print(y.shape)
        # x = np.concatenate([d['frame'] for d in seeds])
        # y = np.concatenate([d[plot_y_value] for d in seeds])
        ax = sns.lineplot(x=x, y=y, n_boot=20, label=label, color=color, linestyle=linestyle, linewidth = linewidth) #TODO might want to change numbers here?

    if y_log_scale:
        plt.yscale('log')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    font_size = font_size
    legend_size = legend_size
    ax.yaxis.offsetText.set_fontsize(font_size)
    ax.xaxis.offsetText.set_fontsize(font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.tick_params(axis='y', which='major', labelsize=font_size)
    ax.tick_params(axis='y', which='minor', labelsize=font_size)
    ax.tick_params(axis='x', which='major', labelsize=font_size)
    ax.tick_params(axis='x', which='minor', labelsize=font_size)
    if no_legend:
        ax.get_legend().remove()
    else:
        plt.legend(prop={'size': legend_size}, loc='best')

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.tight_layout()
    if save_folder is not None:
        if not os.path.isdir(save_folder):
            path = Path(save_folder)
            path.mkdir(parents=True)
        plt.savefig(os.path.join(save_folder, save_name + '.png'))
        plt.close()
    else:
        plt.show()

    #  pretanh pretanh_max  critic_loss critic_q1 actor_logprob
    # if plot_extra_y:
    #     for y_type in [
    #         # 'fps',
    #         # 'total_time',
    #         # 'encoder_loss',
    #         'pretanh',
    #         # 'pretanh_max',
    #         # 'critic_q1',
    #         # 'critic_loss',
    #                    # 'encoder_loss'
    #                    ]:
    #         try:
    #             for variant in variants_to_plot:
    #                 color = variant2colors[variant] if variant2colors is not None else None
    #                 dash = variant2dashes[variant] if variant2dashes is not None else None
    #                 if dash:
    #                     linestyle = 'dashed'
    #                 else:
    #                     linestyle = None
    #                 label = variant2labels[variant] if variant2labels is not None else variant
    #
    #                 seeds = variant2seeds_train[variant]
    #
    #                 x = combine_data_in_seeds(seeds, 'frame', skip=10)
    #                 y = combine_data_in_seeds(seeds, y_type, skip=10, smooth=smooth)
    #
    #                 print(x.shape)
    #
    #                 sns.lineplot(x=x, y=y, n_boot=5, label=label, linestyle=linestyle, color=color) #TODO might want to change numbers here?
    #             plt.ylabel(y_type)
    #             plt.tight_layout()
    #             plt.show()
    #         except Exception as e:
    #             print(e)

def plot_hyper_sensitivity(base_path, variants_base, categories, plot_y_value='success_rate', fig_size=[10, 7], variant_to_label=None,
                           smooth=1, save_folder=None, save_name='', data_skip=1, start_frame=0, end_frame=1000000, verbose=True,
                           seed_max=50, envs=('door', 'relocate')):
    sns.set(rc={'figure.figsize': fig_size})
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    variants_to_plot = []
    variant2info = {}
    for e in envs:
        for i, v in enumerate(variants_base):
            variant = v + '_' + e
            variants_to_plot.append(variant)
            variant2info[variant] = {'score':[], 'cat':categories[i], 'env':e}

    if not isinstance(base_path, list):
        base_path = [base_path,]
    for p in base_path:
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                seed_string_index = folder_name.rfind('s')
                name_no_seed = folder_name[:seed_string_index-1]
                seed_value = int(folder_name[seed_string_index+1:])
                if seed_value > seed_max:
                    continue

                if name_no_seed in variants_to_plot:
                    try:
                        # train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        # variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except:
                        print(name_no_seed, 'empty')

    plot_y_value = 'success_rate' # TODO generalize later, and for now we use average success rate over entire training
    def get_performance_value_from_seeds_for_hyper_sensitivity(seeds):
        # for each seed, get a single value
        score_values = []
        for d in seeds:
            frame_array = d['frame']
            idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
            vals_to_use = d[plot_y_value][idxs].reshape(-1)
            average_score = vals_to_use.mean()
            score_values.append(average_score)
        return score_values

    for variant in variant2seeds_eval:
        variant2info[variant]['score'] = get_performance_value_from_seeds_for_hyper_sensitivity(variant2seeds_eval[variant])

    sns.set_theme(style="darkgrid")
    # now we have them in variant2scores

    df = {'cat':[], 'score':[], 'env':[]}
    for variant in variant2info.keys():
        info = variant2info[variant]
        scores, cat, env = info['score'], info['cat'], info['env']
        for score in scores:
            df['cat'].append(cat)
            df['score'].append(score)
            df['env'].append(env)

    df = pandas.DataFrame(df)
    palette = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:gray']

    have_legend = True
    if have_legend:
        ax = sns.stripplot(data=df, x="cat", y="score", size=30, hue='env', marker='D',
                           edgecolor="gray", alpha=.25, jitter=False, palette=palette)
        # ax = sns.pointplot(data=df, x="cat", y="score", ci="sd", hue='env', capsize=.2, linewidth = 5, palette=palette)
        ax = sns.pointplot(data=df, x="cat", y="score", ci="sd", hue='env', capsize=.2, palette=palette)
        ax.get_legend().remove()

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=palette[0], lw=7, label='Door'),
                           Line2D([0], [0], color=palette[1], lw=7, label='hammer'),
                           Line2D([0], [0], color=palette[2], lw=7, label='Pen'),
                           Line2D([0], [0], color=palette[3], lw=7, label='Relocate'),
                           ]

        # Create the figure
        ax.legend(handles=legend_elements, loc='lower left', prop={'size': 40})
    else:
        ax = sns.pointplot(data=df, x="cat", y="score", ci="sd", hue='env', capsize=.2, linewidth = 5, palette=palette)
        ax = sns.stripplot(data=df, x="cat", y="score", size=30, hue='env', marker='D',
                           edgecolor="gray", alpha=.25, jitter=False, palette=palette)
        ax.get_legend().remove()

    sns.color_palette(('tab:blue', 'tab:orange', 'tab:red'))
    ax.set_ylim([-0.02, 1.02])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    # ax.xaxis.label.set_size(20)
    # ax.yaxis.label.set_size(20)
    ax.tick_params(axis='y', which='major', labelsize=32)
    ax.tick_params(axis='y', which='minor', labelsize=28)
    ax.tick_params(axis='x', which='major', labelsize=44)
    ax.tick_params(axis='x', which='minor', labelsize=40)

    plt.tight_layout()
    if save_folder is not None:
        if not os.path.isdir(save_folder):
            path = Path(save_folder)
            path.mkdir(parents=True)
        save_to = os.path.join(save_folder, save_name + '.png')
        plt.savefig(save_to)
        if verbose:
            print('saved to:', save_to)
        plt.close()
    else:
        plt.show()

"""
variants_to_plot: each entry is like tl_conv4_resnet18_64channel_ftnone_cup_catch_s0 
so the env name is also there, if we want to for example plot 
so we need to aggregate several variants together under a label name 

sth like: 

label2variants = {
'none' : [tl_conv4_resnet18_64channel_ftnone_cup_catch, tl_conv4_resnet18_32channel_ftnone_cup_catch],
'all' : [tl_conv4_resnet18_64channel_ftall_cup_catch, tl_conv4_resnet18_32channel_ftall_cup_catch],
}

This function will create one plot, and each curve in the plot is aggregated over multiple variants... (each variant over multiple seeds)

"""
# TODO currently just do performance plot, will generalize to others later...
def plot_aggregate(paths, label2variants, plot_y_value='episode_reward', plot_extra_y=False, extra_train_y_values=[], fig_size=[16, 9],
         variant2dashes=None, variant2colors=None, variant2labels=None, smooth=1, save_folder=None, save_name='',):
    sns.set(rc={'figure.figsize': fig_size})
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    all_variants = set()
    for label, variants in label2variants.items():
        for variant in variants:
            all_variants.add(variant)

    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    # first load data
    for p in paths:
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                name_no_seed = folder_name[:-3]
                seed_value = int(folder_name[-1])
                if seed_value >= 5:
                    continue

                if name_no_seed in all_variants: # if we should load the data file
                    try:
                        train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except:
                        print(name_no_seed, 'empty')

    # plot here
    label_to_variants_seeds = {}
    for label, variants in label2variants.items():
        label_to_variants_seeds[label] = []
        for variant in variants:
            seeds = variant2seeds_eval[variant]
            label_to_variants_seeds[label] += seeds

        # now we have all variants and all seeds for a label
        x = combine_data_in_seeds(label_to_variants_seeds[label], 'frame', skip=1)
        y = combine_data_in_seeds(label_to_variants_seeds[label], plot_y_value, skip=1, smooth=smooth)

        sns.lineplot(x=x, y=y, n_boot=20, label=label, color=None, linestyle=None)
    plt.tight_layout()
    if save_folder is not None:
        if not os.path.isdir(save_folder):
            path = Path(save_folder)
            path.mkdir(parents=True)
        plt.savefig(os.path.join(save_folder, save_name + '.png'))
        plt.close()
    else:
        plt.show()

def decide_y_label(plot_y_value):
    d = {'episode_reward': 'Performance',
         'performance': 'Performance',
         'success_rate': 'Success Rate',
         'critic_q1': 'Q Value',
         'critic_loss': 'Q Loss',
         'abs_pretanh': 'Abs Pretanh'
         }
    if plot_y_value in d:
        return d[plot_y_value]
    return plot_y_value

def plot_aggregate_0506(paths, label2variants, plot_y_value='episode_reward', plot_extra_y=False, extra_train_y_values=[],
                        fig_size=[7, 7], linewidth=7, font_size=28, legend_size=28, y_label=None,
                        y_log_scale=False, xmin=0, xmax=4050000, ymin=None, ymax=None, no_legend=False,
         label2colors=None, label2linestyle=None, smooth=1, save_folder=None, save_name='', max_frames='auto',
                        max_seed=3, n_boot=20, drqv2_max_frame='auto', x_label='Frames', drq_humanoid_smooth=True):
    # TODO this is the new aggregate plot function, we will use this to generate better looking plots
    sns.set(rc={'figure.figsize': fig_size})
    # for a number of paths, load all data from these paths, then do a performance comparison
    # assume name ends with _sx, x is the seed number and is < 10
    all_variants = set()
    for label, variants in label2variants.items():
        for variant in variants:
            all_variants.add(variant)

    if max_frames == 'auto':
        max_frames = xmax
    if drqv2_max_frame == 'auto': # only to make drqv2 plot look nice
        drqv2_max_frame = max_frames

    if y_label is None:
        y_label = decide_y_label(plot_y_value)
    variant2seeds_train = {}
    variant2seeds_eval = {}
    variant2score = {}
    variant2scorestd = {}
    # first load data
    if plot_y_value == 'episode_reward' or plot_y_value == 'success_rate' or plot_y_value == 'success_rate_aug':
        use_eval = True
    else:
        use_eval = False
    for p in paths:
        for subdir, dirs, files in os.walk(p):
            if 'eval.csv' in files:
                # folder name is here is typically also the variant name
                folder_name = os.path.basename(os.path.normpath(subdir)) # e.g. drq_nstep3_pt5_0.001_hopper_hop_s1
                eval_file_path = os.path.join(subdir, 'eval.csv')
                train_file_path = os.path.join(subdir, 'train.csv')
                seed_string_index = folder_name.rfind('s')
                name_no_seed = folder_name[:seed_string_index-1]
                seed_value = int(folder_name[seed_string_index+1:])
                if seed_value >= max_seed:
                    continue

                if name_no_seed in all_variants: # if we should load the data file
                    try:
                        if not use_eval:
                            train_data = genfromtxt(train_file_path, dtype=float, delimiter=',', names=True)
                        eval_data = genfromtxt(eval_file_path, dtype=float, delimiter=',', names=True)

                        start_frame = 0
                        end_frame = max_frames if 'drqv2_author' not in name_no_seed else drqv2_max_frame
                        frame_array = eval_data['frame']
                        idxs = np.where(np.logical_and(frame_array >= start_frame, frame_array <= end_frame))
                        eval_data = eval_data[idxs]

                        if drq_humanoid_smooth and 'drqv2_author' in name_no_seed and 'humanoid' in name_no_seed:
                            new_frame = []
                            new_episode_reward = []
                            for i in range(len(eval_data['frame'])):
                                frame = eval_data['frame'][i]
                                prev_reward = eval_data['episode_reward'][i]
                                if i < len(eval_data['frame']) - 1:
                                    next_reward = eval_data['episode_reward'][i+1]
                                else:
                                    next_reward =prev_reward
                                for j in range(5):
                                    this_reward = prev_reward + j/5 * (next_reward - prev_reward)
                                    new_frame.append(int(frame + j * 20000))
                                    new_episode_reward.append(this_reward)

                            # eval_data2 = np.array([new_frame, new_episode_reward], dtype=[('frame', np.int), ('episode_reward',np.float)])
                            # eval_data2= pd.DataFrame({'frame':new_frame, 'episode_reward':new_episode_reward}).astype({'frame':int, 'episode_reward':float}).to_numpy()
                            # eval_data2 = np.array([(frame, epi) for zip(new_frame, new_episode_reward)], dtype=[('frame', np.int), ('episode_reward',np.float)])
                            array1 = []
                            for frame, epi in zip(new_frame, new_episode_reward):
                                array1.append((frame, epi))
                            eval_data = np.array(array1, dtype=[('frame', np.int), ('episode_reward',np.float)])

                            # print(eval_data.dtype)
                            # print(eval_data2.dtype)
                            # print(eval_data)
                            # print(eval_data2)
                            # quit()
                        if name_no_seed not in variant2seeds_train:
                            variant2seeds_train[name_no_seed] = []
                            variant2seeds_eval[name_no_seed] = []
                        if not use_eval:
                            variant2seeds_train[name_no_seed].append(train_data)
                        variant2seeds_eval[name_no_seed].append(eval_data)
                    except Exception as e:
                        print(name_no_seed, 'empty', e)

    # plot here
    label_to_variants_seeds = {}
    for label, variants in label2variants.items():
        color = label2colors[label] if label2colors is not None else None
        linestyle = label2linestyle[label] if label2linestyle is not None else None

        label_to_variants_seeds[label] = []
        for variant in variants:
            if use_eval:
                seeds = variant2seeds_eval[variant]
            else:
                seeds = variant2seeds_train[variant]
            label_to_variants_seeds[label] += seeds

        # now we have all variants and all seeds for a label
        x = combine_data_in_seeds(label_to_variants_seeds[label], 'frame', skip=1)
        y = combine_data_in_seeds(label_to_variants_seeds[label], plot_y_value, skip=1, smooth=smooth)

        ax = sns.lineplot(x=x, y=y, n_boot=n_boot, label=label, color=color, linestyle=linestyle, linewidth = linewidth)

    if y_log_scale:
        plt.yscale('log')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    font_size = font_size
    legend_size = legend_size
    ax.yaxis.offsetText.set_fontsize(font_size)
    ax.xaxis.offsetText.set_fontsize(font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.tick_params(axis='y', which='major', labelsize=font_size)
    ax.tick_params(axis='y', which='minor', labelsize=font_size)
    ax.tick_params(axis='x', which='major', labelsize=font_size)
    ax.tick_params(axis='x', which='minor', labelsize=font_size)
    if no_legend:
        ax.get_legend().remove()
    else:
        plt.legend(prop={'size': legend_size}, loc='best')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.tight_layout()

    # plt.show()

    if save_folder is not None:
        if not os.path.isdir(save_folder):
            path = Path(save_folder)
            path.mkdir(parents=True)
        plt.savefig(os.path.join(save_folder, save_name + '.png'))
        plt.close()
    else:
        plt.show()

"""
MEMO
1. first move the new amulet result folder to the base_dir (maybe have a new folder), then add this folder's name to the list_of_data_dir
2. when use to_plot, we use the name of the folder directly above eval.csv and train.csv, use the first part of the name, with the env name and seed part

drqfreeze100k_curl_m3_nstep3_pt5_0.001_hopper_hop_s1
"""




# for a in ['select-hamster', 'modest-guppy', 'accepted-dragon']:
#     move_data_from_amlt(a, 'double')

# base_dir = 'C:\\Users\\v-wangche\\OneDrive - Microsoft\\d\\9-13'
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'drq-freeze', 'drq-enc', 'curl-freeze']
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'curl-freeze', 'ofeaq', 'drq-freeze', ]
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'curl-freeze', 'drq-freeze', 'drq-enc',]
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'curl-freeze', 'drq-freeze2', 'drq-aq-nopretanh']
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'communal-hog', ]
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'drq-highc', 'curl-freeze', ]
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'drq-highc',]
# list_of_data_dir = ['drq-base', 'drq-base-pretanh', 'daring-salmon',]
# list_of_data_dir = ['fixed1', 'drq-base-pretanh',]
# list_of_data_dir = ['fancy', 'drq-base-pretanh',]
# list_of_data_dir = ['drq-base-pretanh','fancynew', ]
# list_of_data_dir = ['base_easy','fixed1', ]
#
# paths = [os.path.join(base_dir, data_dir) for data_dir in list_of_data_dir]
#
# # hyper_search(paths, before_step=200)
# # quit()
#
# # plot(paths, variants_to_plot=['drq_baseline_nstep3_quadruped_walk', 'drq_nstep3_pt5_0.001_quadruped_walk',
# #                               'drq_baseline_nstep3_finger_turn_hard', 'drq_nstep3_pt5_0.001_finger_turn_hard',
# #                               'drq_baseline_nstep3_hopper_hop', 'drq_nstep3_pt5_0.001_hopper_hop',])
#
# to_plot = [
#     # 'drq_baseline_nstep3',
#     # 'drq_nstep3_pt5_0.001',
#     # 'drqpre100k_curl_m3_nstep3_pt5_0.001',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
#     # 'drqpre100k_pred_m2_nstep3_pt5_0.001',
#     # 'drqfreeze100k_pred_m0_nstep3_pt5_0.001',
#     # 'drq_nstep3_pt5_0.005',
#     # 'drq_nstep3_pt5_0.01',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001',
#     #        'drq_ofeaq_u40_lr0.0001_nstep3_pt5_0.001',
#     #        'drq_ofeaq_u80_lr0.0001_nstep3_pt5_0.001',
#            # 'drq_ofeaq_u40_lr0.0002_nstep3_pt5_0.001',
#            # 'drq_ofeaq_u80_lr0.0002_nstep3_pt5_0.001',
#     # 'drq_ofeaq_u120_lr0.0001_nstep3_pt5_0.001',
#     # 'drq_ofeaq_u120_lr0.0002_nstep3_pt5_0.001',
#     # 'drq_ofeaq_u120_lr0.0003_nstep3_pt5_0.001',
#     # 'drq_ofeaq_u120_lr0.0004_nstep3_pt5_0.001',
#     # 'drqfreeze100k_ofe_m3_nstep3_pt5_0',
#     # 'drqfreeze100k_ofe_m3_nstep3_pt5_0.001',
#     # 'drq_ofe_m3_nstep3_pt5_0',
#     # 'drq_ofe_m3_nstep3_pt5_0.001'
#     # 'drqfreeze100k_ofe_m1_affalse_nstep3_pt5_0.001',
#     # 'drq_ofeaq_u40_lr0.0001_nstep3_pt5_0',
#     # 'drq_ofeaq_u120_lr0.0001_nstep3_pt5_0',
#
#     #
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc32_ec0',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc32_ec1',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc32_ec2',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc32_ec3',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc2_ec0',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc2_ec1',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc2_ec2',
#     # 'drqfreeze100k_curl_m3_nstep3_pt5_0.001_ftrue_nc2_ec3',
#
#     # 'drq_nstep3_pt5_0.001_numc2',
#     # 'drq_nstep3_pt5_0.001_numc4',
#     # 'drq_nstep3_pt5_0.001_numc8',
#     # 'drq_nstep3_pt5_0.001_numc16',
#     # 'drq_nstep3_pt5_0.001_numc32',
#     # 'drq_nstep3_pt5_0.001_numc64',
#     # 'drq_nstep3_pt5_0.001_numc96',
#     # 'drq_nstep3_pt5_0.001_numc128',
#
#     'drq_nstep3_pt5_0.001_ftrue_numc2',
#     # 'drq_nstep3_pt5_0.001_ftrue_numc16',
#     'drq_nstep3_pt5_0.001_ftrue_numc32',
#     # 'drq_nstep3_pt5_0.001_ftrue_numc64',
#     # 'drq_nstep3_pt5_0.001_ftrue_numc96',
#     'drq_nstep3_pt5_0.001_ftrue_numc128',
#     # 'drq_nstep3_pt5_0.001_ftrue_numc192',
#
#     'drq_nec_nstep3_pt5_0.001_ftrue_numc2',
#     # 'drq_nec_nstep3_pt5_0.001_ftrue_numc16',
#     'drq_nec_nstep3_pt5_0.001_ftrue_numc32',
#     # 'drq_nec_nstep3_pt5_0.001_ftrue_numc64',
#     'drq_nec_nstep3_pt5_0.001_ftrue_numc128'
#
#     # 'drq_nstep3_pt5_0.001_ffalse_numc4',
#     # # 'drq_nstep3_pt5_0.001_ffalse_numc16',
#     # 'drq_nstep3_pt5_0.001_ffalse_numc32',
#     # 'drq_nstep3_pt5_0.001_ffalse_numc64',
#     # 'drq_nstep3_pt5_0.001_ffalse_numc96',
#     # 'drq_nstep3_pt5_0.001_ffalse_numc128',
#     # 'drq_nstep3_pt5_0.001_ffalse_numc192',
#
# # 'drq_nstep3_pt5_0.001',
# #     'drq_fancy_pretanh_nstep3',
# #     'drq_fancy_nstep3',
#
#     # 'drq_nstep3_pt5_0.001'
#
#
#
# # 'drq_fancy_pretanh_nstep3',
# # 'drq_fancy5_pretanh_nstep3',
#
# ]

# drq_ofeaq_u40_lr0.0002_nstep3_pt5_0.001_hopper_hop_s2

# for env in [
#     # 'quadruped_walk',
#     # 'hopper_hop',
#     # 'finger_turn_hard',
#
# "cartpole_swingup", "cheetah_run", "walker_walk",
# ]:
#     plot(paths, variants_to_plot=[
#         v + '_' + env for v in to_plot
#                                   ],
#          plot_extra_y=True)
#
# quit()
