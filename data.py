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
from data_utils.three_num import TASKS_THREE_NUMS
from data_utils.anli import TASKS_ANLI

TASKS = {**TASKS_THREE_NUMS, **TASKS_TWO_NUMS, **TASKS_ONE_NUM, **TASKS_ANLI}


def get_data(args, task_name: str = 'add_two', n_shots: int = 1, train_split_frac: float = None):
    """

    Params
    ------
    dset: huggingface dataset
    check_answer_func: func
        returns boolean when a string semantically matches the description of a task
    description: str
        string brief description of the task
    train_split: float
        fraction of data to use for training
        Note: if specified, returns tuple of (dset_train, dset_test) instead of single dset_train
    args.max_dset_size: int
        Data (even for NLI datasets) will be truncated to have at most this many rows
    """
    d = defaultdict(list)
    rng = np.random.default_rng(12345)
    # 2nd rng to not break compatibility with earlier data
    rng2 = np.random.default_rng(13)
    task = TASKS[task_name]

    if task_name not in TASKS or task_name == 'SUFFIXES':
        raise Exception(f'{task_name} not in list of supported task names: ' +
                        str(TASKS.keys()) + ' or is "SUFFIXES"')

    # synthetic math task
    if task_name in TASKS_ONE_NUM.keys() \
            or task_name in TASKS_TWO_NUMS.keys() \
            or task_name in TASKS_THREE_NUMS.keys():
        template = task['prompt_template_funcs'][args.template_num_task_phrasing]
        if task_name in TASKS_ONE_NUM.keys():
            num_inputs = 1
        elif task_name in TASKS_TWO_NUMS.keys():
            num_inputs = 2
        elif task_name in TASKS_THREE_NUMS.keys():
            num_inputs = 3

        # dont make unnecessarily big if we're just repeating point
        actual_max_dset_size = min(
            pow(args.max_digit, num_inputs), args.max_dset_size)
        for i in range(actual_max_dset_size):
            # when there are very few possibilities, stratify to use them all
            if args.max_digit == 10 and num_inputs <= 2:
                num1 = i // 10
                num2 = i % 10
            else:
                num1 = rng.integers(low=0, high=args.max_digit)
                num2 = rng.integers(low=0, high=args.max_digit)
                num3 = rng2.integers(low=0, high=args.max_digit)

            gen_func = task['gen_func']
            if num_inputs == 1:
                x, y = template(num2, gen_func)
            elif num_inputs == 2:
                x, y = template(num1, num2, gen_func)
            elif num_inputs == 3:
                x, y = template(num1, num2, num3, gen_func)

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
            all_shots = df.sample(n=n_shots, replace=False)
            d2['text'].append(''.join(all_shots['text'].values))
            #
            last_input = all_shots.tail(n=1)['input'].values[0]
            d2['input'].append(
                ''.join(all_shots['text'].values[:-1]) + last_input)
            d2['last_input'].append(last_input)
            #
            last_output = all_shots.tail(n=1)['output'].values[0]
            d2['output'].append(last_output)
            #

        df = pd.DataFrame.from_dict(d2)
        # shuffle rows

        df = df.sample(n=args.max_dset_size, replace=False)
    # print(df.shape[0], 'max_digit', args.max_digit, 'dset_size', args.max_dset_size, actual_max_dset_size)
    # print(df.head())
    # trim max size (should already be controlled)
    df = df.iloc[:args.max_dset_size]

    # return check answer func
    check_answer_func = TASKS[task_name]['check_answer_func']
    if isinstance(check_answer_func, str):
        check_answer_func_re = re.compile(
            check_answer_func, re.IGNORECASE).search

        def check_answer_func(x): return bool(check_answer_func_re(x))

    # set up task descr
    descr = task['description']
    if not descr.endswith(' '):
        descr += ' '

    if train_split_frac:
        n_train = int(df.shape[0] * train_split_frac)
        dset_train = Dataset.from_pandas(df.iloc[:n_train])
        dset_test = Dataset.from_pandas(df.iloc[n_train:])
        return (dset_train, dset_test), check_answer_func, descr
    else:
        dset = Dataset.from_pandas(df)
        return dset, check_answer_func, descr


def get_init_suffix(args) -> List:
    # Note: don't change the order of these (higher ones should be better)
    """Note: questions should end with 2 newlines, so can directly start suffix.
    """
    if args.use_generic_query:
        return 'To get the answer from the input, return'
    if args.task_name in TASKS_TWO_NUMS.keys():
        init_suffixes = TASKS_TWO_NUMS['SUFFIXES'][args.task_name]
    elif args.task_name in TASKS_ONE_NUM.keys():
        init_suffixes = TASKS_ONE_NUM['SUFFIXES'][args.task_name]
    elif args.task_name in TASKS_ANLI.keys():
        init_suffixes = TASKS_ANLI['SUFFIXES'][args.task_name]
    return init_suffixes[args.template_num_init_string]
