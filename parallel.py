import argparse
import logging
import os
import pickle as pkl
import random
import string
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset
from parallelformers import parallelize
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

import data
import train_prefix
import train_suffix
import utils

device_count = torch.cuda.device_count()
device = 'cpu' if device_count == 0 else 'cuda'

def model_to_device(model):
    if device_count == 0:
        return model
    elif device_count == 1:
        return model.to(device)
    elif device_count > 1:
        parallelize(model, num_gpus=device_count, fp16=False, verbose='detail')
        return model

def inputs_to_device(inputs):
    if device_count == 0 or device_count > 1:
        return inputs
    elif device_count == 1:
        return inputs.to(device)
