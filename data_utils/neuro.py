import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj
import pickle as pkl

NEURO_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'neuro_processed')


def fetch_data():
    # voxel_best_num = int(task_name_neuro.split('_')[-1])
    top_words = pkl.load(
        open(oj(NEURO_PROCESSED_DIR, 'best_voxels_top_words_10000_voxels.pkl'), 'rb'))
    return top_words['top_words'] #[voxel_best_num]
    