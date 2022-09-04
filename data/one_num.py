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
from . import data_utils

"""Note: all templates should be "stackable" so that they work in the multi-shot setting.
Only 2 newlines will be added between them.
Don't change the order of these (higher should be better).
"""
PROMPT_TEMPLATE_ONE_NUM = [
    lambda num1, g: (
        f'Given the input x is {num1}, the output f(x)', f' {g(num1)}.\n\n'),

    lambda num1, g: (
        f'Given the input x is {num1}, the output is', f' {g(num1)}.\n\n'),

    lambda num1, g: (
        f'Given the input number {num1}, the function output is', f' {g(num1)}.\n\n'),

    lambda num1, g: (
        f'Input: {num1}\n', f'Function output: {g(num1)}\n\n'),

    lambda num1, g: (
        f'The input is {num1}.', f' The function output is {g(num1)}\n\n'),
]
SUFFIXES_ONE_NUM = [
        # "The function mapping the input to the output is",
        "The function f(x) returns the",
        "To compute the answer from the input number x, return",
        "To compute the answer f(x) from the input number x, return",
        # "To find the output, take the number in the question and use the",
        # "To get the answer, take the number in the question and",
        "To calculate the answer, take the input and",
        "The relationship between the number in the question and the answer is:",
        "To get the answer,",
]
TASKS_ONE_NUM = {
    'square_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': r'square|(mult.*self)|(prod.*self)|x\s*\*\s*x|pow\(x,\s*2\)',
        'gen_func': lambda x: x * x
    },
    'exp_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': r'exp|e\s*^\s*x|e\s*to the\s*x',
        'gen_func': lambda x: np.exp(x).round(2)
    },
    'prime_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': r'prime',
        'gen_func': lambda x: data_utils.prime_n(x)
    },
    'double_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': r'two|double|2',
        'gen_func': lambda x: data_utils.prime_n(x)
    },

    # too hard
    'fibonacci_one': {
        'prompt_template_funcs': PROMPT_TEMPLATE_ONE_NUM,
        'check_answer_func': r'fib',
        'gen_func': lambda x: data_utils.fib_n(x)
    },

    'SUFFIXES': SUFFIXES_ONE_NUM,
}
