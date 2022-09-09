from functools import partial
import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import trange
import torch.nn
from data_utils import data_funcs
from data_utils.one_num import TASKS_ONE_NUM
from data_utils.two_num import TASKS_TWO_NUMS
from data_utils.anli import TASKS_ANLI

TASKS = {**TASKS_TWO_NUMS, **TASKS_ONE_NUM, **TASKS_ANLI}

def get_data(args, task_name: str = 'add_two', n_shots: int = 1):
    """Return
    dset: huggingface dataset
    check_answer_func: func
        returns boolean when a string semantically matches the description of a task
    description: str
        string brief description of the task
    """
    d = defaultdict(list)
    rng = np.random.default_rng(12345)
    task = TASKS[task_name]
    actual_max_dset_size = args.max_dset_size
    
    # synthetic math task
    if task_name in TASKS_ONE_NUM.keys() or task_name in TASKS_TWO_NUMS.keys():
        template = task['prompt_template_funcs'][args.template_num_task_phrasing]
        if task_name in TASKS_ONE_NUM.keys():
            num_inputs = 1
        if task_name in TASKS_TWO_NUMS.keys():
            num_inputs = 2
        # dont make unnecessarily big if we're just repeating point
        actual_max_dset_size = min(pow(args.max_digit, num_inputs), args.max_dset_size)
        for i in range(actual_max_dset_size):
            # when there are very few possibilities, stratify to use them all
            if args.max_digit == 10 and num_inputs <= 2:
                num1 = i // 10
                num2 = i % 10
            else:
                num1 = rng.integers(low=0, high=args.max_digit)
                num2 = rng.integers(low=0, high=args.max_digit)

            gen_func = task['gen_func']
            if num_inputs == 1:
                x, y = template(num2, gen_func)
            elif num_inputs == 2:
                x, y = template(num1, num2, gen_func)

            d['text'].append(x + y)
            d['input'].append(x)
            d['output'].append(y)
        df = pd.DataFrame.from_dict(d)

    # NLI task
    elif task_name in TASKS_ANLI.keys():
        df = task['gen_func'](task_name)


    
    
    df = df.sample(n=min(df.shape[0], args.max_dset_size),
                   replace=False)  # shuffle rows

    # reprocess for the multi-shot setting
    if n_shots > 1:
        logging.debug('Note: multi-shot is not supported by prefix search')
        d2 = defaultdict(list)
        for i in range(args.max_dset_size):
            s = ''.join(df.sample(n=n_shots, replace=False)['text'].values)
            d2['text'].append(s)
        df = pd.DataFrame.from_dict(d2)
        # shuffle rows
        df = df.sample(n=actual_max_dset_size, replace=False)

    # print(df.shape[0], 'max_digit', args.max_digit, 'dset_size', args.max_dset_size, actual_max_dset_size)
    # print(df.head())
    # trim max size (should already be controlled)
    df = df.iloc[:args.max_dset_size]
    dset = Dataset.from_pandas(df)

    # return check answer func
    check_answer_func = TASKS[task_name]['check_answer_func']
    if isinstance(check_answer_func, str):
        check_answer_func_re = re.compile(check_answer_func, re.IGNORECASE).search
        check_answer_func = lambda x: bool(check_answer_func_re(x))
    return dset, check_answer_func, task['description']


"""Note: questions should end with 2 newlines, so can directly start suffix.
"""
def get_init_suffix(args) -> List:
    # Note: don't change the order of these (higher ones should be better)
    if args.task_name in TASKS_TWO_NUMS.keys():
       init_suffixes = TASKS_TWO_NUMS['SUFFIXES']
    elif args.task_name in TASKS_ONE_NUM.keys():
        init_suffixes = TASKS_ONE_NUM['SUFFIXES']
    elif args.task_name in TASKS_ANLI.keys():
        init_suffixes = TASKS_ANLI['SUFFIXES'] 
    return init_suffixes[args.template_num_init_string]


if __name__ == '__main__':
    print('\n################Lots of available dsets############\n')
    for task_key in TASKS:
        if not task_key == 'SUFFIXES':
            print(task_key, '->', TASKS[task_key]['description'] + '\n') #, TASKS[task_key]['description'])


    print('\n\n################Lets look at some examples############\n')
    class fake_args:
        template_num_task_phrasing = 0
        max_dset_size = 1000
        max_digit=10
    task_name = 'multiply_two'

    args = fake_args()
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=3)
    print('Example 3-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    args.max_digit = 100
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=100)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name='fibonacci_one', n_shots=1)
    print('Example fibonacci_one 1-shot (max_digit=10)',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    print('\n\n################Lets look at an ANLI dataset############\n')
    task_name = 'task1147_country_currency'
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print(f'Example {task_name} 1-shot',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=3)
    print(f'Example {task_name} 3-shot',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    print('\n\n################Lets look at how answers are checked############\n')
    task_name = 'add_two'
    task = TASKS[task_name]
    _, check_answer_func, descr = get_data(args, task_name=task_name)
    print('checking func', check_answer_func, 'for', task_name)
    for s in ['add', 'take the nums and add', 'test', ' sum', 'Add']:
        print(repr(s), check_answer_func(s))
