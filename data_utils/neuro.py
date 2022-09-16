import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj
import pickle as pkl

NEURO_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'neuro_processed')


def fetch_data(task_name_neuro: str = 'neuro_0'):
    voxel_best_num = int(task_name_neuro.split('_')[-1])
    top_words = pkl.load(
        open('neuro_processed/best_voxels_top_words_10000_voxels.pkl', 'rb'))
    return top_words[voxel_best_num]


TASKS_NEURO = {
    **{
        f'neuro_{i}': {'check_answer_func': None}
        for i in range(1000)
    },
    'SUFFIXES': ['Given the input country, the answer is the country\'s'],
}
ks = list(TASKS_NEURO.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_NEURO[k]['description'] = 'Find the similarity between the top words'
        TASKS_NEURO[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_NEURO)
