from model_utils import prompt_classification
import data
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


task_names_math_one = ['square_one', 'exp_one', 'double_one', 'fibonacci_one']
task_names_math_two = ['add_two', 'multiply_two', 'divide_two', 'subtract_two',
                       'max_two', 'first_two']
task_names_math_three = ['add_three', 'multiply_three', 'max_three', 'first_three']
task_names_anli = ['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
                   'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
                   'task092_check_prime_classification', 'task088_identify_typo_verification',
                   'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql'
                   ]



######################## ACTUAL HYPERPARAMS ################################
checkpoints = ['EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B']
max_digit = 10 # If this is not 10, only math gets run!
TASK_SETTINGS = {
    'one_digit_all': task_names_math_one + task_names_math_two + task_names_anli,
    'double_digit_two_nums': task_names_math_two,
    'one_digit_three_nums': task_names_math_three,
}
task_key = 'one_digit_three_nums'
tasks = TASK_SETTINGS[task_key]
############################################################################



# prepare the args
batch_sizes = {
    'gpt2-medium': 32,
    'EleutherAI/gpt-j-6B': 8,
    'EleutherAI/gpt-neo-2.7B': 16,
    'EleutherAI/gpt-neox-20b': 1,
}
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

# , , 'gpt2-xl', 'EleutherAI/gpt-neox-20b']:
for checkpoint in checkpoints:
    d = defaultdict(list)
    print('loading', checkpoint)
    model = prompt_classification.create_model(checkpoint)
    print('calculating accs...')
    for task_name in tqdm(tasks):
        for n_shots in [1, 5, 10]:
            for prompt in ['', 'manual']:
                args.task_name = task_name
                args.n_shots = n_shots
                args.max_digit = max_digit
                (dset, dset_test), check_answer_func, descr = data.get_data(
                    args, args.task_name, n_shots=args.n_shots,
                    train_split_frac=args.train_split_frac
                )
                d['checkpoint'].append(checkpoint)
                d['prompt'].append(prompt)
                d['task_name'].append(task_name)
                d['n_shots'].append(n_shots)
                if prompt == 'manual':
                    prompt_actual = descr
                else:
                    prompt_actual = prompt
                d['prompt_actual'].append(prompt_actual)
                batch_size = batch_sizes.get(checkpoint, 16)
                if task_name == 'task107_splash_question_to_sql':
                    batch_size = max(1, batch_size//4)
                loss, acc = prompt_classification.test_model_on_task_with_prefix(
                    dset=dset, model=model, prefix=prompt_actual, multi_token=True, verbose=False,
                    batch_size=batch_size,
                )
                d['acc'].append(acc)
            if task_key == 'one_digit_all':
                save_name = f'results/generalization_acc/baseline_accs_{checkpoint.replace("/", "___")}___nshots={n_shots}.pkl'
            elif task_key == 'double_digit_two_nums':
                save_name = f'results/generalization_acc/baseline_accs_{checkpoint.replace("/", "___")}___nshots={n_shots}___maxdigit={max_digit}.pkl'
            elif task_key == 'one_digit_three_nums':
                save_name = f'results/generalization_acc/baseline_accs_{checkpoint.replace("/", "___")}___nshots={n_shots}___three_nums.pkl'
            pkl.dump(d, open(save_name, 'wb'))
