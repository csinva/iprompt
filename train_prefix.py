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
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

import data
import utils

def get_init_prefix(model, dataloader, tokenizer, wte, device) -> List:
    """
    """
    prefix_str = ["x the following two numbers: "]
    prefix_inputs = tokenizer(prefix_str, return_tensors="pt").to(device)
    prefix_emb = wte.forward(prefix_inputs['input_ids'])
    prefix_emb = torch.nn.Parameter(prefix_emb).to(device)
    return prefix_emb

def train_prefix(args, r, model, dataloader, save_dir, tokenizer):
    """Gradient-based optimization of the prefix
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # extract out embedding
    wte = model._modules['transformer'].wte.to(device)

    # get embedding of a chosen prefix string as nn.Parameter
    prefix_emb = get_init_prefix(
        model, dataloader, tokenizer, wte, device)

    # optimizer
    optim = torch.optim.Adam([prefix_emb], lr=args.lr_prefix)

    # run training loop
    for epoch in range(args.n_epochs_prefix):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x_text = batch['input']
            y_text = batch['output']
            full_text = [x_text[i] + y_text[i] for i in range(len(x_text))]
            # print(full_text)
            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            # print(ex_inputs)
            ex_embs = wte.forward(ex_inputs['input_ids'].to(
                device)).to(device)

            # concatenate prefix + example
            emb = torch.cat((prefix_emb.repeat(ex_embs.shape[0], 1, 1),
                            ex_embs), dim=1)

            # go through model
            outputs = model(inputs_embeds=emb)

            # calculate loss
            # currently this calculates loss only on the answer token
            idxs_correct = tokenizer(y_text, return_tensors='pt')[
                'input_ids'].to(device)
            assert idxs_correct.nelement(
            ) == args.batch_size, 'For now assume that answer is a single token'
            # (batch_size, seq_len, vocab_size)

            last_token_logits = outputs['logits'][:, -1, :]
            log_probs = torch.gather(last_token_logits, 1, idxs_correct)

            # accumulate gradients in this batch
            loss = -1 * log_probs.mean()  # minimize prob answer being wrong

            loss.backward()

        # save stuff
        r['embs'].append(prefix_emb.detach().cpu().numpy())
        r['grads'].append(prefix_emb.grad.detach().cpu().numpy())
        r['losses'].append(loss.item())
        utils.save(args, save_dir, r, epoch=epoch)
        # print('losses', loss)

        # optimize
        optim.step()
        optim.zero_grad()