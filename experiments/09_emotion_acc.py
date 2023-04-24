from iprompt import prompt_classification
import iprompt.data as data
import numpy as np
import torch
from torch import nn
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd
from types import SimpleNamespace
from datasets import Dataset
from os.path import join as oj
import pickle as pkl
import os
from os.path import dirname
repo_dir = dirname(os.path.abspath(__file__))
results_acc_dir = oj(repo_dir, 'results', 'generalization_acc')
os.makedirs(results_acc_dir, exist_ok=True)


task_names_math_one = ['square_one', 'exp_one', 'double_one', 'fibonacci_one']
task_names_math_two = ['add_two', 'multiply_two', 'divide_two', 'subtract_two',
                       'max_two', 'first_two']
task_names_math_three = ['add_three',
                         'multiply_three', 'max_three', 'first_three']
task_names_anli = ['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
                   'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
                   'task092_check_prime_classification', 'task088_identify_typo_verification',
                   'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql'
                   ]
task_names_sentiment = ['ffb_train', 'imdb_train',
                        'rt_train', 'sst2_train']  # , 'tweets_train']


######################## ACTUAL HYPERPARAMS ################################
checkpoints_test = [
    # 'gpt2',
    # 'facebook/opt-2.7b',
    # 'EleutherAI/gpt-j-6B',
    # 'facebook/opt-6.7b',
    # 'EleutherAI/gpt-neo-2.7B',
    # 'EleutherAI/gpt-neox-20b',
    # 'facebook/opt-66b',
    'gpt3',
]
TASK_SETTINGS = {
    'emotion': {
        'n_shots': [1],
        'multi_token': True,
    }
}
PROMPTS_EMOTION = [
    " Classify whether the tweet's emotion is sad, happy, love, anger, fear, or surprise:", # is 'emotion': {0: 'Sad', 1: 'Happ', 2: 'Love', 3: 'Ang', 4: 'Fear', 5: 'Surpris'},'
    " What emotion is in this sentence? Choose from sad, happy, love, anger, fear, or surprise. Emotion:",
    ' What feeling is present in this tweet? Choose from sad, happy, love, anger, fear, or surprise:',
    ' This tweet contains the emotion',
    ' The emotion of this tweet is',
]
task_key = 'emotion'
prompt = PROMPTS_EMOTION[2]

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

parallelize = False
args = fake_args()
np.random.seed(args.seed)
settings = TASK_SETTINGS[task_key]
if 'max_digit' in settings:
    args.max_digit = settings['max_digit']
# args.train_split_frac = settings['train_split_frac']
if 'max_dset_size' in settings:
    args.max_dset_size = settings['max_dset_size']
if 'prompt_seed' in settings:
    prompt_seed = settings['prompt_seed']
else:
    prompt_seed = 1
if 'multi_token' in settings:
    multi_token = settings['multi_token']
else:
    multi_token = True

for checkpoint in checkpoints_test:
    print('loading', checkpoint)
    model = prompt_classification.create_model(checkpoint, parallelize)
    print('calculating accs...')
    # which task to test on
    for task_name_test in ['emotion_train']: #tqdm(settings['task_names']):

        # which task to get prompt from
        if 'task_names_prompt' in settings:
            task_names_prompt = settings['task_names_prompt']
        else:
            task_names_prompt = [task_name_test]
        for task_name_prompt in task_names_prompt:
            d = defaultdict(list)
            for n_shots in settings['n_shots']:

                # load the data we are going to test on
                if task_name_test in task_names_sentiment:
                    # remap key for sentiment (train and test on different dsets)
                    if task_name_test.endswith('train'):
                        args.task_name = task_name_test.replace(
                            'train', 'test')
                    else:
                        args.task_name = task_name_test
                else:
                    args.task_name = task_name_test
                args.n_shots = n_shots
                data_kwargs = dict(
                    task_name=args.task_name, n_shots=args.n_shots,
                    train_split_frac=args.train_split_frac,
                    max_dset_size=args.max_dset_size,
                    template_num_task_phrasing=args.template_num_task_phrasing,
                    max_digit=args.max_digit,
                )
                if args.train_split_frac:
                    (_, dset_test), _, descr = data.get_data(**data_kwargs)
                else:
                    dset_test, _, descr = data.get_data(**data_kwargs)

                # load the prompt we are going to use
                prompt_actual = prompt

                # calculate acc
                batch_size = batch_sizes.get(checkpoint, 16)
                if checkpoint == 'gpt3':
                    acc = prompt_classification.test_gpt_model_on_task_with_prefix(
                        dset=dset_test, prefix=prompt_actual, verbose=True, multi_token=multi_token,
                        use_lower=True
                    )
                else:
                    _, acc = prompt_classification.test_model_on_task_with_prefix(
                        dset=dset_test, model=model, prefix=prompt_actual, multi_token=multi_token, verbose=True,
                        batch_size=batch_size, use_lower=True,# prefix_before_input=False,
                    )

                # save stuff
                d['checkpoint'].append(checkpoint)
                d['task_name'].append(task_name_test)
                d['n_shots'].append(n_shots)
                d['max_digit'].append(args.max_digit)
                # d['train_split_frac'].append(args.train_split_frac)
                d['prompt_actual'].append(prompt_actual)
                d['acc'].append(acc)
                d['task_name_prompt'].append(task_name_prompt)
                print('acc', acc)

                baseline_acc_dir = oj(results_acc_dir, 'baseline_accs')
                ckpt = checkpoint.replace("/", "___")
                save_name = oj(baseline_acc_dir, f'baseline_accs_{ckpt}___nshots={n_shots}.pkl')
                os.makedirs(baseline_acc_dir, exist_ok=True)
                pkl.dump(d, open(save_name, 'wb'))
