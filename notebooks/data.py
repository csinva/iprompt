import logging
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import trange


def get_data(max_digit=1000, template_idx=-1, n_shots: int = 1, max_dset_size=10000):
    TEMPLATES = [  # note, these should be stackable in the multi-shot setting
        lambda num1, num2: (
            f' the numbers {num1} and {num2} and the answer is', f' {num1 + num2}'),

        lambda num1, num2: (
            f'{num1} {num2}', f' {num1 + num2}\n'),

        lambda num1, num2: (
            f'The input is {num1} {num2}', f' The answer is {num1 + num2}\n\n'),

        lambda num1, num2: (
            f'Question: {num1} {num2}\n', f'Answer: {num1 + num2}\n\n'),
    ]

    d = defaultdict(list)
    np.random.seed(13)
    template = TEMPLATES[template_idx]
    for num1 in trange(max_digit, desc="creating data", leave=False):
        for num2 in range(min(max_digit, max_dset_size//max_digit)):
            x, y = template(num1, num2)
            d["text"].append(x + y)
            d['input'].append(x)
            d['output'].append(y)
    df = pd.DataFrame.from_dict(d)
    df = df.sample(n=df.shape[0], replace=False)  # shuffle rows

    """
    for i in range(max_digit * max_digit):   
        num1 = np.random.randint(0, max_digit)
        num2 = np.random.randint(0, max_digit)
        d['input'].append(f'{num1:03} {num2:03}')
        d['output'].append(f' {num1 + num2:04}')
    """

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
    return dset


if __name__ == '__main__':
    df = get_data(max_digit=100, n_shots=3, max_dset_size=1000)
    for i in range(3):
        print(df[i]['text'], end='<--------------------\n')
