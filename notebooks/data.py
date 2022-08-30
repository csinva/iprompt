from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import trange

def get_data(max_digit=1000):
    d = defaultdict(list)
    np.random.seed(13)
    for num1 in trange(max_digit, desc="creating data", leave=False):
        for num2 in range(max_digit):
            d['input'].append(f'{num1} {num2}')
            d['output'].append(f' {num1 + num2}')
    """
    for i in range(max_digit * max_digit):   
        num1 = np.random.randint(0, max_digit)
        num2 = np.random.randint(0, max_digit)
        d['input'].append(f'{num1:03} {num2:03}')
        d['output'].append(f' {num1 + num2:04}')
    """

    df = pd.DataFrame.from_dict(d)
    # print(df.head())
    df = df.sample(frac=1) # shuffle rows
    # print(df.head())
    dset = Dataset.from_pandas(df)
    return dset