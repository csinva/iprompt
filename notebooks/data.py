from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict

def get_data(N=1000):
    N = 1000
    d = defaultdict(list)
    np.random.seed(13)
    for i in range(N):
        num1 = np.random.randint(0, 100)
        num2 = np.random.randint(0, 100)
        d['input'].append(f'{num1} {num2}')
        d['output'].append(f' {num1 + num2}')
    df = pd.DataFrame.from_dict(d)
    dset = Dataset.from_pandas(df)
    return dset