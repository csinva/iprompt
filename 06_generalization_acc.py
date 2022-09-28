from model_utils import prompt_classification
import data
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
task_names_math_three = ['add_three', 'multiply_three', 'max_three', 'first_three']
task_names_anli = ['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
                   'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
                   'task092_check_prime_classification', 'task088_identify_typo_verification',
                   'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql'
                   ]
task_names_sentiment = ['ffb_train', 'imdb_train', 'rt_train', 'sst2_train', 'tweets_train']


######################## ACTUAL HYPERPARAMS ################################
checkpoints_test = [
    # 'gpt2',
    # 'facebook/opt-2.7b',
    'EleutherAI/gpt-j-6B',
    # 'facebook/opt-6.7b',
    # 'EleutherAI/gpt-neo-2.7B',
    # 'EleutherAI/gpt-neox-20b',
    # 'facebook/opt-66b',
    # 'gpt3',
]
TASK_SETTINGS = {
    'one_digit_all': {
        'task_names': task_names_math_one + task_names_math_two + task_names_anli,
        'max_digit': 10,
        'n_shots': [1, 5, 10],
        'prompt_types': ['', 'manual'],
        'train_split_frac': None,
    },
    'double_digit_two_nums': {
        'task_names': task_names_math_two,
        'max_digit': 100,
        'n_shots': [1, 5, 10],
        'prompt_types': ['', 'manual'],
        'train_split_frac': None,
    },
    'one_digit_three_nums': {
        'task_names': task_names_math_three,
        'max_digit': 10,
        'n_shots': [1, 5, 10],
        'prompt_types': ['', 'manual'],
        'train_split_frac': None,
    },
    'sweep_in_distr_math': {
        'task_names': task_names_math_one + task_names_math_two, # + task_names_anli,
        'max_digit': 10,
        'n_shots': [1],
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual', 'suffix'],
        'train_split_frac': 0.75,
    },
    'sweep_double_digit_math': {
        'task_names': task_names_math_two,
        'max_digit': 100,
        'n_shots': [1],
        'prompt_types': ['suffix', 'autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
    },
    'sweep_one_digit_three_nums_math': {
        'task_names': task_names_math_three,
        'max_digit': 10,
        'n_shots': [1],
        'prompt_types': ['suffix', 'autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
    },
    'sweep_in_distr_anli': {
        'task_names': task_names_anli,
        'n_shots': [1],
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual', 'suffix'],
        'train_split_frac': 0.75,
    },    
    'sweep_sentiment_1': {
        'task_names': task_names_sentiment,
        'n_shots': [1],
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
        'prompt_seed': 1,
        'multi_token': False,
    },
    'sweep_sentiment_2': {
        'task_names': task_names_sentiment,
        'n_shots': [1],
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
        'prompt_seed': 2,
        'multi_token': False,
    },
    'sweep_sentiment_3': {
        'task_names': task_names_sentiment,
        'n_shots': [1],
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
        'prompt_seed': 3,
        'multi_token': False,
    },
    'sweep_sentiment_cross_distr': {
        'task_names': task_names_sentiment,
        'task_names_prompt': task_names_sentiment, # get prompts from a different distr than testing
        'max_dset_size': 100000, # use the whole dset
        'n_shots': [1],
        'prompt_seed': 1,
        'prompt_types': ['autoprompt', 'iprompt', '', 'manual'], 
        'train_split_frac': None,
        'multi_token': False,
    }
}

# task_keys = ['sweep_in_distr_math', 'sweep_in_distr_anli']
# task_keys = ['sweep_sentiment']
# task_keys = ['sweep_sentiment_2', 'sweep_sentiment_3', 'sweep_sentiment_1']
task_keys = ['sweep_sentiment_cross_distr']
# task_keys = ['sweep_in_distr_math']
# task_keys = ['sweep_double_digit_math', 'sweep_one_digit_three_nums_math']
# task_keys = ['sweep_in_distr_anli']
for task_key in task_keys:

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
    args.train_split_frac = settings['train_split_frac']
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
    prompts_saved = pkl.load(open(oj(results_acc_dir, 'prompts_all.pkl'), 'rb'))
    prompts_sent = pkl.load(open(oj(repo_dir, 'results/autoprompt_sentiment/prompts.pkl'), 'rb')).reset_index()
    

    for checkpoint in checkpoints_test:
        print('loading', checkpoint)
        model = prompt_classification.create_model(checkpoint, parallelize)
        print('calculating accs...')
        # which task to test on
        for task_name_test in tqdm(settings['task_names']):

            # which task to get prompt from
            if 'task_names_prompt' in settings:
                task_names_prompt = settings['task_names_prompt']
            else:
                task_names_prompt = [task_name_test]
            for task_name_prompt in task_names_prompt:
                for prompt_type in settings['prompt_types']:
                    d = defaultdict(list)
                    for n_shots in settings['n_shots']:
                        
                        # load the data we are going to test on
                        if task_name_test in task_names_sentiment:
                            # remap key for sentiment (train and test on different dsets)
                            if task_name_test.endswith('train'):
                                args.task_name = task_name_test.replace('train', 'test')
                            else:
                                args.task_name = task_name_test
                        else:
                            args.task_name = task_name_test
                        args.n_shots = n_shots
                        if args.train_split_frac:
                            (_, dset_test), _, descr = data.get_data(
                                args, args.task_name, n_shots=args.n_shots,
                                train_split_frac=args.train_split_frac)
                        else:
                            dset_test, _, descr = data.get_data(
                                    args, args.task_name, n_shots=args.n_shots,
                                    train_split_frac=args.train_split_frac) 

                        # load the prompt we are going to use
                        if prompt_type == 'manual':
                            prompt_actual = descr
                        elif prompt_type in ['autoprompt', 'iprompt', 'suffix']:
                            if task_name_prompt.endswith('three'):
                                task_name_prompt = task_name_prompt.replace('three', 'two')
                            elif task_name_prompt in task_names_sentiment:
                                # remap some keys for sentiment
                                if prompt_type == 'iprompt':
                                    pt = 'genetic'
                                else:
                                    pt = prompt_type
                                prompt_actual = prompts_sent[
                                    (prompts_sent.task_name == task_name_prompt) * \
                                        (prompts_sent.model_cls == pt) * \
                                            (prompts_sent.seed == prompt_seed)
                                ]['prefixes'].iloc[0]
                            else:
                                # get saved prompt
                                prompt_actual = prompts_saved.loc[task_name_prompt][prompt_type]
                        elif prompt_type == '':
                            prompt_actual = prompt_type
                        
                        # calculate acc
                        batch_size = batch_sizes.get(checkpoint, 16)
                        if task_name_test == 'task107_splash_question_to_sql':
                            batch_size = max(1, batch_size//4)
                        if checkpoint == 'gpt3':
                            acc = prompt_classification.test_gpt_model_on_task_with_prefix(
                                dset=dset_test, prefix=prompt_actual, verbose=True, multi_token=multi_token,
                            )
                        else:
                            _, acc = prompt_classification.test_model_on_task_with_prefix(
                                dset=dset_test, model=model, prefix=prompt_actual, multi_token=multi_token, verbose=False,
                                batch_size=batch_size,
                            )

                        # save stuff
                        d['checkpoint'].append(checkpoint)
                        d['prompt'].append(prompt_type)
                        d['task_name'].append(task_name_test)
                        d['n_shots'].append(n_shots)
                        d['max_digit'].append(args.max_digit)
                        d['train_split_frac'].append(args.train_split_frac)
                        d['prompt_actual'].append(prompt_actual)
                        d['acc'].append(acc)
                        d['task_name_prompt'].append(task_name_prompt)
                        d['prompt_seed'].append(prompt_seed)

                    baseline_acc_dir = oj(results_acc_dir, 'baseline_accs')
                    ckpt = checkpoint.replace("/", "___")
                    if task_key == 'one_digit_all':
                        save_name = oj(baseline_acc_dir, f'baseline_accs_{ckpt}___nshots={n_shots}.pkl')
                    elif task_key == 'double_digit_two_nums':
                        save_name = oj(baseline_acc_dir, f'baseline_accs_{ckpt}___nshots={n_shots}___maxdigit={args.max_digit}.pkl')
                    elif task_key == 'one_digit_three_nums':
                        save_name = oj(baseline_acc_dir, f'baseline_accs_{ckpt}___nshots={n_shots}___three_nums.pkl')
                    elif task_key.startswith('sweep'):
                        save_name = f'accs_sweep/accs_{ckpt}__{task_key}__{task_name_test}__prompt_type={prompt_type}'
                        if not task_name_prompt == task_name_test:
                            save_name += f'___{task_name_prompt}'
                        if prompt_seed > 1:
                            save_name += f'___ps={prompt_seed}'
                        save_name = oj(
                            results_acc_dir,
                            f'{save_name}.pkl'
                        )
                        print('save_name', save_name)
                    pkl.dump(d, open(save_name, 'wb'))
