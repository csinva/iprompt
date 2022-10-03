import json
from copy import deepcopy
import os
import pandas as pd
from os.path import dirname
from os.path import join as oj
import pickle as pkl
import numpy as np
from datasets import Dataset

NEURO_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'neuro_processed')


def fetch_data(n_words=None):
    # voxel_best_num = int(task_name_neuro.split('_')[-1])
    top_words = pkl.load(
        open(oj(NEURO_PROCESSED_DIR, 'best_voxels_top_words_10000_voxels.pkl'), 'rb'))
    if n_words is None:
        return top_words['top_words']  # [voxel_best_num]
    else:
        return top_words['top_words'][:, :n_words]


def fetch_meta():
    top_words = pkl.load(
        open(oj(NEURO_PROCESSED_DIR, 'voxels_metadata.pkl'), 'rb'))
    return top_words


def remap_scores_best_to_scores_all(scores_best_voxels, corrsort):
    """We are only looking at the best voxels (in the order of corrsort).
    Before plotting, we need to map back.
    """
    scores_all_voxels = np.zeros(corrsort.size)
    for i in range(scores_best_voxels.size):
        vox_num = corrsort[i]
        scores_all_voxels[vox_num] = scores_best_voxels[i]
    return scores_all_voxels


def fetch_permuted_word_list_for_voxel(num_shuffles=15, vox_num=0, n_words=15):
    rng = np.random.default_rng(77)
    word_list = fetch_data(n_words=n_words)[vox_num]
    word_lists_shuffled = []
    for i in range(num_shuffles):
        word_lists_shuffled.append(deepcopy(word_list))
        rng.shuffle(word_list)
    words_list_shuffled_np = np.array(word_lists_shuffled)

    def join_each(arr, add_leading_space=False):
        if add_leading_space:
            return [' ' + ' '.join(x) for x in arr]
        else:
            return [' '.join(x) for x in arr]
    print(words_list_shuffled_np.shape)
    d = {
        'text': join_each(word_lists_shuffled),
        'input': join_each(words_list_shuffled_np[:, :n_words//2]),
        'output': join_each(words_list_shuffled_np[:, n_words//2:], add_leading_space=True),
    }
    return Dataset.from_pandas(pd.DataFrame.from_dict(d))