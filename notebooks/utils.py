import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datasets import Dataset
import data
import pickle as pkl

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
    w_embed = trans.wte.weight # vocab_size, embed_dim
    vocab_size = w_embed.shape[0]
    embed_size = w_embed.shape[1]

    # invert for unembedding
    unemb_linear = nn.Linear(in_features=embed_size, out_features=vocab_size, bias=False)
    pinv = torch.linalg.pinv(w_embed)
    unemb_linear.weight = nn.Parameter(pinv.T)
    
    pkl.dump(unemb_linear, open(fname, 'wb'))
    return unemb_linear


def save(args, save_dir, r, epoch=None):
    if epoch is None or (epoch % args.epoch_save_interval == 0):
        os.makedirs(save_dir, exist_ok=True)
        pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))