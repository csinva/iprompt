import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import trange
import torch.nn
import data_utils


def get_data(args, task_name: str = 'add_two', max_digit: int = 1000,
             n_shots: int = 1):

    d = defaultdict(list)
    rng = np.random.default_rng(12345)
    task = TASKS[task_name]
    template = task['prompt_template_funcs'][args.template_num_task_phrasing]
    if task_name in TASKS_ONE_NUM.keys():
        num_inputs = 1
    if task_name in TASKS_TWO_NUMS.keys():
        num_inputs = 2
    # dont make unnecessarily big if we're just repeating point
    actual_max_dset_size = min(pow(max_digit, num_inputs), args.max_dset_size)
    for i in range(actual_max_dset_size):
        # when there are very few possibilities, stratify to use them all
        if max_digit == 10 and num_inputs <= 2:
            num1 = i // 10
            num2 = i % 10
        else:
            num1 = rng.integers(low=0, high=max_digit)
            num2 = rng.integers(low=0, high=max_digit)

        gen_func = task['gen_func']
        if num_inputs == 1:
            x, y = template(num1, gen_func)
        elif num_inputs == 2:
            x, y = template(num1, num2, gen_func)

        d["text"].append(x + y)
        d['input'].append(x)
        d['output'].append(y)
    df = pd.DataFrame.from_dict(d)
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

    print(df.shape[0], 'max_digit', max_digit, 'dset_size', args.max_dset_size, actual_max_dset_size)
    # print(df.head())
    # trim max size (should already be controlled)
    df = df.iloc[:args.max_dset_size]
    dset = Dataset.from_pandas(df)
    return dset, TASKS[task_name]['check_answer_func']


"""Note: all templates should be "stackable" so that they work in the multi-shot setting.
Nothing will be added between them (so probably should end with 2 newlines).
Don't change the order of these (higher should be better).
"""

PROMPT_TEMPLATE_TWO_NUMS = [
    lambda num1, num2, g: (
        f'Given the input numbers {num1} and {num2}, the answer is', f' {g([num1, num2])}.\n\n'),

    lambda num1, num2, g: (
        f'Inputs: {num1} {num2}\n', f'Answer: {g([num1, num2])}\n\n'),

    lambda num1, num2, g: (
        f'The inputs are {num1} {num2}.', f' The answer is {g([num1, num2])}\n\n'),
]


"""Note: tasks here consist of
(i) a function that returns data given the right inputs
(ii) a function that evaluates whether the output is correct
(iii) a function that computes the output (given a list)
"""
TASKS_TWO_NUMS = {
    'add_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'add|sum').search,
        'gen_func': sum
    },
    'multiply_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'multiply|product').search,
        'gen_func': np.prod
    },
    'divide_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'divide|into').search,
        'gen_func': lambda l: f'{l[0]}/{l[1]}'
    },
    'subtract_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'subtract|difference').search,
        'gen_func': lambda l: l[0] - l[1]
    },
    'max_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'max|large|greate|big').search,
        'gen_func': max
    },

    # this one finds solutions like "subtract from the first"
    'first_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'first|begin|original').search,
        'gen_func': lambda x: x[0]
    },
}

"""Note: all templates should be "stackable" so that they work in the multi-shot setting.
Only 2 newlines will be added between them.
Don't change the order of these (higher should be better).
"""
PROMPT_TEMPLATE_ONE_NUM = [
    lambda num1, g: (
        f'Given the input number {num1}, the function output is', f' {g(num1)}.\n\n'),

    lambda num1, g: (
        f'Input: {num1}\n', f'Function output: {g(num1)}\n\n'),

    lambda num1, g: (
        f'The input is {num1}.', f' The function output is {g(num1)}\n\n'),
]
TASKS_ONE_NUM = {
    'fibonacci_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': re.compile(r'fib').search,
        'gen_func': lambda x: data_utils.fib(x)
    },
    'square_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': re.compile(r'square|(mult.*self)|(prod.*self)').search,
        'gen_func': lambda x: x * x
    }
}

TASKS = {**TASKS_TWO_NUMS, **TASKS_ONE_NUM}


"""Note: questions should end with 2 newlines, so can directly start suffix.
"""
def get_init_suffix(args) -> List:

    # Note: don't change the order of these (higher ones should be better)
    if args.task_name in TASKS_TWO_NUMS.keys():
        init_suffixes = [
            "To get the answer, take the numbers in the question and",
            "To compute the answer from the inputs",
            "To calculate the answer, take the two inputs and",
            "The relationship between the numbers in the question and the answer is:",
            "To get the answer,",
        ]
    elif args.task_name in TASKS_ONE_NUM.keys():
        init_suffixes = [
            "To compute the answer from the input number,",
            "The function mapping the input to the output is",
            "To find the output, take the number in the question and use the",
            # "To get the answer, take the number in the question and",
            "To calculate the answer, take the input and",
            "The relationship between the number in the question and the answer is:",
            "To get the answer,",
        ]
    return init_suffixes[args.template_num_init_string]


if __name__ == '__main__':
    class fake_args:
        template_num_task_phrasing = 0
        max_dset_size = 1000
    task_name = 'multiply_two'

    args = fake_args()
    dset, check_answer_func = get_data(
        args, max_digit=10, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func = get_data(
        args, max_digit=100, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=100)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func = get_data(
        args, max_digit=10, task_name=task_name, n_shots=3)
    print('Example 3-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func = get_data(
        args, max_digit=20, task_name='fibonacci_one', n_shots=1)
    print('Example fibonacci_one one-shot (max_digit=10)',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

"""
    task = TASKS['first_two']
    func = task['check_answer_func']
    print('func', func)
    print(bool(func('add')), func('take the numbers and add'),
          bool(func('test')), func(' sum'))

    gen_func = task['gen_func']
    template = task['prompt_template_funcs'][0]
    print('gen_func', gen_func, 'template', template)
    print(template(3, 4, gen_func))
"""
