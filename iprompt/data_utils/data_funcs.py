import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import trange
import torch.nn
from iprompt.data_utils.one_num import TASKS_ONE_NUM
from iprompt.data_utils.two_num import TASKS_TWO_NUMS
from iprompt.data_utils.three_num import TASKS_THREE_NUMS

def get_task_dataframe(task, task_name: str, max_digit: int, max_dset_size: int,
template_num_task_phrasing: int, rng, rng2):
    template = task['prompt_template_funcs'][template_num_task_phrasing]
    if task_name in TASKS_ONE_NUM.keys():
        num_inputs = 1
    elif task_name in TASKS_TWO_NUMS.keys():
        num_inputs = 2
    elif task_name in TASKS_THREE_NUMS.keys():
        num_inputs = 3

    # dont make unnecessarily big if we're just repeating point
    actual_max_dset_size = min(
        pow(max_digit, num_inputs), max_dset_size)
    d = defaultdict(list)
    for i in range(actual_max_dset_size):
        # when there are very few possibilities, stratify to use them all
        if max_digit == 10 and num_inputs <= 2:
            num1 = i // 10
            num2 = i % 10
        else:
            num1 = rng.integers(low=0, high=max_digit)
            num2 = rng.integers(low=0, high=max_digit)
            num3 = rng2.integers(low=0, high=max_digit)

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
    return pd.DataFrame.from_dict(d)

def fib_n(n):
    a = 0
    b = 1
    if n == 0:
        return a
    elif n == 1:
        return b
    else:
        for i in range(2,n+1):
            c = a + b
            a = b
            b = c
        return b


def prime_n(n):
    # initial prime number list
    prime_list = [2]
    # first number to test if prime
    num = 3
    # keep generating primes until we get to the nth one
    while len(prime_list) < n:

        # check if num is divisible by any prime before it
        for p in prime_list:
            # if there is no remainder dividing the number
            # then the number is not a prime
            if num % p == 0:
                # break to stop testing more numbers, we know it's not a prime
                break

        # if it is a prime, then add it to the list
        # after a for loop, else runs if the "break" command has not been given
        else:
            # append to prime list
            prime_list.append(num)

        # same optimization you had, don't check even numbers
        num += 2

    # return the last prime number generated
    return prime_list[-1]