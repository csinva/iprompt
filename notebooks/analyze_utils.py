import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
from datasets import Dataset
from os.path import join as oj
import pickle as pkl
import json
import os


def load_results_and_cache(results_dir: str, save_file: str = 'r.pkl') -> pd.DataFrame:
    """Suffix script stores results_final.pkl instead of results.json
    """
    dir_names = sorted([fname
                        for fname in os.listdir(results_dir)
                        if os.path.isdir(oj(results_dir, fname))
                        and os.path.exists(oj(results_dir, fname, 'results_final.pkl'))
                        ])
    results_list = []
    for dir_name in tqdm(dir_names):
        try:
            ser = pd.Series(
                pkl.load(open(oj(results_dir, dir_name, 'results_final.pkl'), "rb")))
            # only keep scalar-valued vars
            ser_scalar = ser[~ser.apply(
                lambda x: isinstance(x, list)).values.astype(bool)]
            results_list.append(ser_scalar)
        except Exception as e:
            print('skipping', dir_name)

    r = pd.concat(results_list, axis=1).T.infer_objects()
    r.to_pickle(oj(results_dir, save_file))
    return r


def load_results_and_cache_prefix_json(results_dir: str, save_file: str = 'r.pkl') -> pd.DataFrame:
    """Prefix script stores results.json instead of results_final.pkl
    """
    dir_names = sorted([fname
                        for fname in os.listdir(results_dir)
                        if os.path.isdir(oj(results_dir, fname))
                        and os.path.exists(oj(results_dir, fname, 'results.json'))
                        ])
    dfs = []
    for dir_name in tqdm(dir_names):
        try:
            json_filename = oj(results_dir, dir_name, 'results.json')
            json_dict = json.load(open(json_filename, 'r'))
            # backwards compatibility with prev unneeded key
            del json_dict['task_name_list']
            df = pd.DataFrame.from_dict(json_dict)
            df['json_filename'] = json_filename

            # if we computed accuracy, reorder by that metric.
            rerank = df['do_reranking'].tolist()[0]
            if rerank:
                df = df.sort_values(by='accs', ascending=False).reset_index()

            # get index of first answer, which will be nan if there isn't one (if all
            # answers were wrong).
            first_answer_idx = (
                df.index[df['prefixes__check_answer_func']]).min()
            if pd.isna(first_answer_idx):
                df['final_answer_full'] = np.NaN
                df['final_answer_pos_initial_token'] = float('inf')
            else:
                df['final_answer_full'] = df.prefixes.iloc[first_answer_idx]
                df['final_answer_pos_initial_token'] = first_answer_idx

            dfs.append(df)
        except Exception as e:
            print("e:", e)
            print('skipping', dir_name)

    r = pd.concat(dfs, axis=0)
    r.to_pickle(oj(results_dir, save_file))
    return r


def postprocess_results(r):
    """
    # drop some things to make it easier to see
    cols = r.columns
    cols_to_drop = [k for k in cols
                    if k.endswith('prefix') or k.endswith('init') # or k.startswith('use')
                    ]
    cols_to_drop += ['epoch_save_interval', 'batch_size']
    r = r.drop(columns=cols_to_drop)
    """

    # some keys that might not be set
    KEY_DEFAULTS = {
        'final_model_queries': 1,
        'final_answer_added': '',
        'final_num_suffixes_checked': 1,
        'final_answer_depth': 1,
    }
    for k in KEY_DEFAULTS.keys():
        if not k in r.columns:
            r[k] = KEY_DEFAULTS[k]

    # print(r.keys())
    if 'final_answer_full' in r.columns:
        r['final_answer_found'] = (~r['final_answer_full'].isna()).astype(int)
    else:
        r['final_answer_found'] = 0

    r['use_single_query'] = (
        r['use_single_query']
        .astype(bool)
        .map({True: 'Single-query',
              False: 'Avg suffix'})
    )

    # add metrics
    metric_keys = []
    for i in [3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        metric_key = f'Recall @ {i} suffixes'
        r[metric_key] = (r['final_answer_pos_initial_token'] <= i)
        metric_keys.append(metric_key)
    r['reciprocal_rank'] = r['final_answer_pos_initial_token'].map(lambda n: 1/(n+1))
    if 'final_num_suffixes_checked' in r.columns:
        r['reciprocal_rank_multi'] = r['final_num_suffixes_checked'].map(lambda n: 1/(n+1)) # use the results found by beam search
    return r


def num_suffixes_checked_tab(tab, metric_key='final_num_suffixes_checked'):
    return (tab
            # mean over templates, task_name)
            .groupby(['checkpoint', 'n_shots', 'use_single_query'])[[metric_key]]
            .mean()
            .reset_index()
            )


LEGEND_REMAP = {
    'Single-query': 'Single-output sampling, suffix',
    'Avg suffix': 'Average-output sampling, suffix',
}
# light to dark
# blues ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
# grays ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
# reds ['#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73']
# greens ['#348ba6', '#38aaac', '#55caad', '#a1dfb9']
COLORS = OrderedDict({
    'Suffix, single-output decoding (1-Ex.)': '#d9d9d9',
    'Suffix, single-output decoding (5-Ex.)': '#969696',
    'Suffix, single-output decoding (10-Ex.)': '#525252',
    'Suffix, average-output decoding (1-Ex.)': '#9ecae1',
    'Suffix, average-output decoding (5-Ex.)': '#4292c6',
    'Suffix, average-output decoding (10-Ex.)': '#084594',
    ############################################################
    'Prefix, single-output (1-Ex.)': '#d9d9d9',
    'Prefix, single-output (5-Ex.)': '#bdbdbd',
    'Prefix, single-output with reranking (1-Ex.)': '#969696',
    'Prefix, single-output with reranking (5-Ex.)': '#525252',
    'Prefix, average-output (1-Ex.)': '#f69c73',
    'Prefix, average-output (5-Ex.)': '#e83f3f',
    'Prefix, average-output with reranking (1-Ex.)': '#a11a5b',
    'Prefix, average-output with reranking (5-Ex.)': '#4c1d4b',

}.items())
SORTED_HUE_NAMES = list(COLORS.keys())


YLABS = {
    'final_num_suffixes_checked': 'Number of suffixes checked before finding correct answer\n(lower is better)',
    'final_answer_pos_initial_token': 'Rank of correct suffix (lower is better)',
    'reciprocal_rank': 'Reciprocal rank (higher is better)',
    'reciprocal_rank_multi': 'Reciprocal rank (higher is better)',
}


def get_hue_order(legend_names):
    for hue in legend_names.unique():
        assert hue in SORTED_HUE_NAMES, hue + \
            ' not in ' + str(SORTED_HUE_NAMES)
    return [k for k in SORTED_HUE_NAMES if k in legend_names.unique()]


def plot_tab(tab: pd.DataFrame, metric_key: str, title: str, add_legend: bool = True, legend_on_side=True):

    # reformat legend
    if add_legend:
        tab['legend'] = tab['use_single_query'].map(
            LEGEND_REMAP) + ' (' + tab['n_shots'].astype(str) + '-Ex.)'

    # sort the plot
    hue_order = get_hue_order(tab['legend'])



    CHECKPOINT_RENAME = {
        'gpt2-medium': 'GPT-2 (345M)',
        'gpt2-large': 'GPT-2 (774M)',
        'gpt2-xl': 'GPT-2 (1.5B)',
        'EleutherAI/gpt-neo-2.7B': 'GPT-Neo (2.7B)',
        'EleutherAI/gpt-j-6B': 'GPT-J (6B)',
        'EleutherAI/gpt-neox-20b': 'GPT-Neo (20B)',
    }
    SORTED_MODEL_NAMES = [
        'gpt2-medium', 'gpt2-large', 'gpt2-xl',
        'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b',
    ]
    for checkpoint in tab['checkpoint'].unique():
        assert checkpoint in SORTED_MODEL_NAMES, checkpoint + \
            ' not in ' + str(SORTED_MODEL_NAMES)
    order = [CHECKPOINT_RENAME[k] for k in SORTED_MODEL_NAMES if k in tab['checkpoint'].unique()]

    
    # print(tab['checkpoint'].map(CHECKPOINT_RENAME))

    # make plot
    # ax = sns.barplot(x=tab['checkpoint'], y=metric_key, hue='legend', hue_order=hue_order, order=order,
    #  data=tab, palette=COLORS)  # data=tab[tab['n_shots'] == 1])
    ax = sns.barplot(x=tab['checkpoint'].map(CHECKPOINT_RENAME).tolist(),
                     y=tab[metric_key], hue=tab['legend'], hue_order=hue_order, order=order, palette=COLORS)  # data=tab[tab['n_shots'] == 1])
    # ax = sns.barplot(x=tab['checkpoint'].map(CHECKPOINT_RENAME).tolist(),
                    #  y=tab[metric_key], hue=tab['legend'], palette=COLORS)
    plt.xlabel('Model name')
    plt.ylabel(YLABS.get(metric_key, metric_key))
    # plt.title(title, fontsize='medium')

    # remove legend title
    handles, labels = ax.get_legend_handles_labels()
    # , labelcolor='linecolor')
    if legend_on_side:
        ax.legend(handles=handles[:], labels=labels[:], bbox_to_anchor=(1, 0.95), frameon=False)
    else:
        plt.legend()
    #   loc='center left')

    plt.tight_layout()
    # plt.show()
