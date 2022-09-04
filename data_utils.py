import logging
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import trange
import torch.nn

def fib(n):
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