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

def get_unembedding(model):
    trans = model._modules['transformer']
    w_embed = trans.wte.weight # vocab_size, embed_dim
    vocab_size = w_embed.shape[0]
    embed_size = w_embed.shape[1]

    # invert for unembedding
    unemb_linear = nn.Linear(in_features=embed_size, out_features=vocab_size, bias=False)
    pinv = torch.linalg.pinv(w_embed)
    unemb_linear.weight = nn.Parameter(pinv.T)
    return unemb_linear
