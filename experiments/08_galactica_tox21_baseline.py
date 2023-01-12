import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datasets import Dataset
from os.path import join as oj
import pickle as pkl
import os
import sys
import iprompt.data
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from iprompt.data import TASKS_GALACTICA
import transformers
from imodelsx import explain_dataset_iprompt, get_add_two_numbers_dataset

if __name__ == '__main__':
    
    # hyperparams
    for seed in range(36):
        task_name = f'tox21_baseline'
        n_max = 100
        save_dir = '/home/chansingh/iprompt/experiments/results'

        # get task
        # task = TASKS_GALACTICA[task_name]
        
        # get data
        # df = task['gen_func']()
        input_strings = [' '] * n_max #df['input'].values[:n_max]
        output_strings = ['yes'] * n_max #df['output'].values[:n_max]

        # explain the relationship between the inputs and outputs
        # with a natural-language prompt string
        prompts, metadata = explain_dataset_iprompt(
            input_strings=input_strings,
            output_strings=output_strings,
            # checkpoint='EleutherAI/gpt-j-6B', # which language model to use
            checkpoint="facebook/galactica-6.7b", # which language model to use
            num_learned_tokens=6, # how long of a prompt to learn
            n_shots=5, # shots per example
            n_epochs=1, # how many epochs to search
            verbose=0, # how much to print
            llm_float16=True, # whether to load the model in float_16
            max_n_datapoints=64,
            seed=seed,
        )

        os.makedirs(save_dir, exist_ok=True)
        pkl.dump(metadata, open(oj(save_dir, f'{task_name}_{n_max}_{seed}.pkl'), 'wb'))




