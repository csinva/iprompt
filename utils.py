import os
import random
import string
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
import pandas as pd
from os.path import join as oj
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datasets import Dataset
import data
import pickle as pkl
import json
import logging
from dict_hash import sha256


def get_unembedding(checkpoint):
    """Get unembedding layer for first continuous vector
    This is needed to take gradients wrt the input text
    """
    checkpoint_clean = checkpoint.lower().replace('/', '___')
    fname = f'../data/preprocessed/unembed_{checkpoint_clean}.pkl'
    if os.path.exists(fname):
        return pkl.load(open(fname, 'rb'))

    # get the embedding from the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    trans = model._modules['transformer']
    w_embed = trans.wte.weight  # vocab_size, embed_dim
    vocab_size = w_embed.shape[0]
    embed_size = w_embed.shape[1]

    # invert for unembedding
    unemb_linear = nn.Linear(in_features=embed_size,
                             out_features=vocab_size, bias=False)
    pinv = torch.linalg.pinv(w_embed)
    unemb_linear.weight = nn.Parameter(pinv.T)

    pkl.dump(unemb_linear, open(fname, 'wb'))
    return unemb_linear


def save_json(args={}, save_dir='results', fname='params.json', r={}):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, fname), 'w') as f:
        if isinstance(args, dict):
            json.dump({**args, **r}, f, indent=4)
        else:
            json.dump({**vars(args), **r}, f, indent=4)


def save(args, save_dir, r, epoch=None, final=False, params=True):
    os.makedirs(save_dir, exist_ok=True)
    if final:
        pkl.dump(r, open(os.path.join(save_dir, 'results_final.pkl'), 'wb'))
    elif epoch is None or (epoch % args.epoch_save_interval == 0):
        pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))


def check_cached(save_dir_unique_hash, args, args_ignore_for_caching, parser, save_dir) -> bool:
    """Check if this configuration has already been run.
    Breaks if parser changes (e.g. changing default values of cmd-line args)
    """
    if not os.path.exists(save_dir):
        return False
    exp_dirs = [d for d in os.listdir(save_dir)
                if os.path.isdir(oj(save_dir, d))]
    args_dict = vars(args)
    defaults = vars(parser.parse_args([]))
    non_default_args = {k: args_dict[k] for k in args_dict.keys()
                        if not k in args_ignore_for_caching and
                        not args_dict[k] == defaults[k]
                        }

    logging.info('checking for cached run...')
    for exp_dir in tqdm(exp_dirs):
        try:
            if exp_dir.startswith(save_dir_unique_hash):

                # probably matched, but to be safe let's check the params
                params_file = oj(save_dir, exp_dir, 'params.json')
                results_final_file = oj(save_dir, exp_dir, 'results_final.pkl')
                if os.path.exists(params_file) and os.path.exists(results_final_file):
                    d = json.load(open(params_file, 'r'))
                    perfect_match = True
                    for k in non_default_args:
                        if not d[k] == non_default_args[k]:
                            perfect_match = False
                    if perfect_match:
                        logging.info('match found at ' + results_final_file)
                        return True
        except:
            pass
    return False


def get_unique_dir_hash(parser, args, args_ignore_for_caching) -> str:
    args = vars(args)
    defaults = vars(parser.parse_args([]))
    non_default_args = {k: args[k] for k in args.keys()
                        if not k in args_ignore_for_caching and
                        not args[k] == defaults[k]}
    return sha256(non_default_args)
