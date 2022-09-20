import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj
import pickle as pkl
import numpy as np

NEURO_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'neuro_processed')


def fetch_data(n_words=None):
    # voxel_best_num = int(task_name_neuro.split('_')[-1])
    top_words = pkl.load(
        open(oj(NEURO_PROCESSED_DIR, 'best_voxels_top_words_10000_voxels.pkl'), 'rb'))
    if n_words is None:
        return top_words['top_words'] #[voxel_best_num]
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