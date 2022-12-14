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
from . import data_funcs

"""Note: all templates should be "stackable" so that they work in the multi-shot setting.
Nothing will be added between them (so probably should end with 2 newlines).
Don't change the order of these (higher should be better).
"""

PROMPT_TEMPLATE_THREE_NUMS = [
    lambda num1, num2, num3, g: (
        f'Given the input numbers {num1}, {num2}, and {num3}, the answer is', f' {g([num1, num2, num3])}.\n\n'),
]
SUFFIXES_THREE_NUMS = {
    'add_three': ["To compute the answer, take the input numbers and"],
    'multiply_three': ["To compute the answer, take the input numbers and"],
    'max_three': ["To compute the answer, take the input numbers and return their"],
    'first_three': ["To compute the answer, return the"],
}
# "To compute the answer f(x) from the input number x, return",
# "To calculate the answer, take the input and",
# "To find the output, take the number in the question and use the",
# "To get the answer, take the number in the question and",
# "The function mapping the input to the output is",
# "The function f(x) returns the",
# "The relationship between the number in the question and the answer is:",
# "To get the answer,",

"""Note: tasks here consist of
(i) a function that returns data given the right inputs
(ii) a function that evaluates whether the output is correct
(iii) a function that computes the output (given a list)
"""
TASKS_THREE_NUMS = {
    'add_three': {
        'prompt_template_funcs': PROMPT_TEMPLATE_THREE_NUMS,
        'check_answer_func': r'add|sum|\+',
        'gen_func': sum,
        'description': "Return the sum of the inputs.",
    },
    'multiply_three': {
        'prompt_template_funcs': PROMPT_TEMPLATE_THREE_NUMS,
        'check_answer_func': r'multiply|product|\*',
        'gen_func': np.prod,
        'description': "Return the product of the inputs.",
    },
    'max_three': {
        'prompt_template_funcs': PROMPT_TEMPLATE_THREE_NUMS,
        'check_answer_func': r'max|large|greate|big',
        'gen_func': max,
        'description': "Return the maximum of the inputs.",
    },
    'first_three': {
        'prompt_template_funcs': PROMPT_TEMPLATE_THREE_NUMS,
        'check_answer_func': r'first|begin|original',
        'gen_func': lambda x: x[0],
        'description': "Return the first of the inputs.",
    },

    'SUFFIXES': SUFFIXES_THREE_NUMS,
}
