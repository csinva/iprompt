import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import trange
import torch.nn

def get_data(task_name='add_two', max_digit=1000, template_idx=-1,
             n_shots: int = 1, max_dset_size=10000):

    d = defaultdict(list)
    rng = np.random.default_rng(12345)
    task = TASKS[task_name]
    template = task['prompt_template_funcs'][template_idx]
    for num1 in trange(max_digit, desc="creating data", leave=False):
        for num2 in range(min(max_digit, max_dset_size//max_digit)):
            gen_func = task['gen_func']
            x, y = template(num1, num2, gen_func)
            d["text"].append(x + y)
            d['input'].append(x)
            d['output'].append(y)
    df = pd.DataFrame.from_dict(d)
    df = df.sample(n=df.shape[0], replace=False)  # shuffle rows

    """
    for i in range(max_digit * max_digit):   
        num1 = rng.randint(0, max_digit)
        num2 = rng.randint(0, max_digit)
        d['input'].append(f'{num1:03} {num2:03}')
        d['output'].append(f' {num1 + num2:04}')
    """

    # reprocess for the multi-shot setting
    if n_shots > 1:
        logging.debug('Note: multi-shot is not supported by prefix search')
        d2 = defaultdict(list)
        for i in range(max_dset_size):
            s = ''.join(df.sample(n=n_shots, replace=False)['text'].values)
            d2['text'].append(s)
        df = pd.DataFrame.from_dict(d2)
        df = df.sample(n=df.shape[0], replace=False)  # shuffle rows

    # print(df.head())

    # print(df.head())
    # trim max size (should already be controlled)
    df = df.iloc[:max_dset_size]
    dset = Dataset.from_pandas(df)
    return dset, TASKS[task_name]['check_answer_func']


"""Note: all templates should be "stackable" so that they work in the multi-shot setting
"""
PROMPT_TEMPLATE_TWO_NUMS = [
    lambda num1, num2, g: (
        f'Given the numbers {num1} and {num2}, the answer is', f' {g([num1, num2])}'),

    lambda num1, num2, g: (
        f'{num1} {num2}', f' {g([num1, num2])}\n'),

    lambda num1, num2, g: (
        f'The input is {num1} {num2}', f' The answer is {g([num1, num2])}\n\n'),

    lambda num1, num2, g: (
        f'Question: {num1} {num2}\n', f'Answer: {g([num1, num2])}\n\n'),
]


"""Note: tasks here consist of
(i) a function that returns data given the right inputs
(ii) a function that evaluates whether the output is correct
(iii) a function that computes the output (given a list)
"""
TASKS = {
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
    'max_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'max|large|greate|big').search,
        'gen_func': max
    },
    'first_two': {
        'prompt_template_funcs': PROMPT_TEMPLATE_TWO_NUMS,
        'check_answer_func': re.compile(r'first|begin|original').search,
        'gen_func': lambda x: x[0]
    },
}

def get_init_prefix(model, dataloader, tokenizer, wte, device) -> List:
    prefix_str = ["x the following two numbers: "]
    prefix_inputs = tokenizer(prefix_str, return_tensors="pt").to(device)
    prefix_emb = wte.forward(prefix_inputs['input_ids'])
    prefix_emb = torch.nn.Parameter(prefix_emb).to(device)
    return prefix_emb

def get_init_suffix(model, dataloader, tokenizer, device) -> List:
    addition_suffixes_manual = [
        "The relationship between the numbers in the question and the answer is:",
        "To get the answer, take the two numbers in the question and",
        "To get the answer,",
        "To get the answer, take the two inputs and",
    ]
    
    return addition_suffixes_manual[-1]



if __name__ == '__main__':
    """
    df = get_data(max_digit=100, n_shots=3, max_dset_size=1000)
    for i in range(3):
        print(df[i]['text'], end='<--------------------\n')
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
