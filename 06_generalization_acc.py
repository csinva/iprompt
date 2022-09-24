import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
from types import SimpleNamespace
from datasets import Dataset
from os.path import join as oj
import pickle as pkl
import os
import dvu
dvu.set_style()
import data
from model_utils import prompt_classification

class fake_args:
    template_num_task_phrasing = 0
    max_dset_size = 1000
    max_digit = 10
    seed = 1
    train_split_frac = 0.75

    # these will be varied
    n_shots = 1
    task_name = 'add_two'
args = fake_args()
np.random.seed(args.seed)


task_names = ['add_two', 'multiply_two', 'divide_two', 'subtract_two',
              'max_two', 'first_two',
              'square_one', 'exp_one', 'double_one', 'fibonacci_one'] + \
    ['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
     'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
     'task092_check_prime_classification', 'task088_identify_typo_verification',
     'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql'
     ]

for checkpoint in ['gpt2-medium', 'EleutherAI/gpt-j-6B', 'gpt2-xl', 'EleutherAI/gpt-neox-20b']:
    d = defaultdict(list)
    print('loading', checkpoint)
    model = prompt_classification.create_model(checkpoint)
    for prompt in ['', 'manual']:
        for task_name in tqdm(task_names):
            for n_shots in [1, 5]: 
                    args.task_name = task_name
                    args.n_shots = n_shots
                    (dset, dset_test), check_answer_func, descr = data.get_data(
                        args, args.task_name, n_shots=args.n_shots, train_split_frac=args.train_split_frac)
                    d['checkpoint'].append(checkpoint)
                    d['prompt'].append(prompt)
                    d['task_name'].append(task_name)
                    d['n_shots'].append(n_shots)
                    if prompt == 'manual':
                        prompt_actual = descr
                    else:
                        prompt_actual = prompt
                    d['prompt_actual'].append(prompt_actual)
                    batch_size = 16
                    if checkpoint == 'EleutherAI/gpt-neox-20b':
                        batch_size = 1
                    loss, acc = prompt_classification.test_model_on_task_with_prefix(
                        dset=dset, model=model, prefix=prompt_actual, multi_token=True, verbose=False,
                    )
                    d['acc'].append(acc)
        pkl.dump(d, open(f'results/generalization_acc/baseline_accs_{checkpoint.replace("/", "___")}.pkl', 'wb'))