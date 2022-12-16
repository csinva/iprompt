import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from iprompt.data_utils import data_funcs
from iprompt.data_utils.one_num import TASKS_ONE_NUM
from iprompt.data_utils.two_num import TASKS_TWO_NUMS
from iprompt.data_utils.three_num import TASKS_THREE_NUMS
from iprompt.data_utils.anli import TASKS_ANLI
from iprompt.data_utils.classification import TASKS_CLASSIFICATION
from iprompt.data_utils.induction import TASKS_INDUCTION
from iprompt.data_utils.d3 import TASKS_D3
from iprompt.data_utils.galactica import TASKS_GALACTICA

TASKS = {
    **TASKS_THREE_NUMS, **TASKS_TWO_NUMS,
    **TASKS_ONE_NUM, **TASKS_ANLI, **TASKS_CLASSIFICATION,
    **TASKS_INDUCTION, **TASKS_D3,
    **TASKS_GALACTICA,
}


def get_data(task_name: str = 'add_two',
             n_shots: int = 1,
             train_split_frac: float = None,
             max_dset_size: int = 10000,
             template_num_task_phrasing: int = 0,
             max_digit: int = 10,
             ):
    """

    Params
    ------
    dset: str
        huggingface dataset name or custom dataset name
    n_shots: int
        number of examples to put in the context (1 for no examples)
    train_split_frac: float
        fraction of data to use for training
        Note: if specified, returns tuple of (dset_train, dset_test) instead of single dset_train
    max_dset_size: int
        maximum number of examples to use (truncate if larger than this)
    template_num_task_phrasing: int
        which template to use for task phrasing
    max_digit: int
        maximum digit to use in the task (if performing synthetic math)

    Returns
    -------
    dsets: HuggingFace Dataset or tuple of HuggingFace Datasets
        if train_split_frac is None, returns single dset_train
        else returns tuple of (dset_train, dset_test)
    check_answer_func: func
        function to check if a string accurately describes this dataset
    descr: str
        string describing the dataset
    """
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
        df = data_funcs.get_task_dataframe(task, task_name, max_digit, max_dset_size,
                                           template_num_task_phrasing, rng, rng2)

    # NLI task, or classification
    else:
        df = task['gen_func'](task_name)

    assert {'text', 'input', 'output'} <= set(
        df.columns), f"got bad columns {df.columns}"

    # Example dataframe row:
    # {'text': 'Given the input numbers 69 and 22, the answer is 91.\n\n',
    # 'input': 'Given the input numbers 69 and 22, the answer is',
    # 'output': ' 91.\n\n'}

    df = df.sample(n=min(df.shape[0], max_dset_size),
                   replace=False)  # shuffle rows

    # reprocess for the multi-shot setting
    if n_shots > 1:
        logging.debug('Note: multi-shot is not supported by prefix search')
        d2 = defaultdict(list)
        for i in range(max_dset_size):
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

        df = df.sample(n=max_dset_size, replace=False)
    # print(df.shape[0], 'max_digit', max_digit, 'dset_size', max_dset_size, actual_max_dset_size)
    # print(df.head())
    # trim max size (should already be controlled)
    df = df.iloc[:max_dset_size]

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


def get_init_suffix(task_name: str, use_generic_query: bool = False, template_num_init_string: int = 0) -> List:
    # Note: don't change the order of these (higher ones should be better)
    """Note: questions should end with 2 newlines, so can directly start suffix.
    """
    if use_generic_query:
        return 'To get the answer from the input, return'
    if task_name in TASKS_TWO_NUMS.keys():
        init_suffixes = TASKS_TWO_NUMS['SUFFIXES'][task_name]
    elif task_name in TASKS_ONE_NUM.keys():
        init_suffixes = TASKS_ONE_NUM['SUFFIXES'][task_name]
    elif task_name in TASKS_THREE_NUMS.keys():
        init_suffixes = TASKS_THREE_NUMS['SUFFIXES'][task_name]
    elif task_name in TASKS_ANLI.keys():
        init_suffixes = TASKS_ANLI['SUFFIXES'][task_name]
    elif task_name in TASKS_CLASSIFICATION.keys():
        init_suffixes = TASKS_CLASSIFICATION['SUFFIXES'][task_name]
    else:
        raise Exception(f'no suffix found for task {task_name}')
    return init_suffixes[template_num_init_string]
