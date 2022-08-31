import os
import random
import string
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datasets import Dataset
import data
import utils
import search

import logging
import pickle as pkl
from torch.utils.data import DataLoader
from datetime import datetime


def train_prefix(args, r, model, wte, dataloader, device, save_dir, search_emb):
    # optimizer
    optim = torch.optim.Adam([search_emb], lr=args.lr)

    # run training loop
    for epoch in range(args.n_epochs):
        for idx, batch in tqdm(enumerate(dataloader)):
            x_text = batch['input']
            y_text = batch['output']
            full_text = [x_text[i] + y_text[i] for i in range(len(x_text))]
            # print(full_text)
            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            # print(ex_inputs)
            ex_embs = wte.forward(ex_inputs['input_ids'].to(
                device)).to(device)

            # concatenate prefix + example
            emb = torch.cat((search_emb.repeat(ex_embs.shape[0], 1, 1),
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
        r['embs'].append(search_emb.detach().cpu().numpy())
        r['grads'].append(search_emb.grad.detach().cpu().numpy())
        r['losses'].append(loss.item())
        utils.save(epoch, args, save_dir, r)
        # print('losses', loss)

        # optimize
        optim.step()
        optim.zero_grad()


def train_suffix(args, r, model, dataloader, device, save_dir):
    """Here we find the suffix which maximizes the likelihood over all examples.
    Not really technically `training` anything (no optimizable parameters, just sampling for a string).
    """
    # optimizer
    # optim = torch.optim.Adam() #[search_emb], lr=args.lr)

    # run training loop
    for epoch in range(args.n_epochs):
        for idx, batch in tqdm(enumerate(dataloader)):
            x_text = batch['input']
            y_text = batch['output']
            full_text = [x_text[i] + y_text[i] for i in range(len(x_text))]
            # print(full_text)
            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            # print(ex_inputs)
            ex_embs = wte.forward(ex_inputs['input_ids'].to(
                device)).to(device)

            # concatenate prefix + example
            emb = torch.cat((search_emb.repeat(ex_embs.shape[0], 1, 1),
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
        # r['embs'].append(search_emb.detach().cpu().numpy())
        # r['grads'].append(search_emb.grad.detach().cpu().numpy())
        r['losses'].append(loss.item())
        utils.save(epoch, args, save_dir, r)
        # print('losses', loss)

        # optimize
        # optim.step()
        # optim.zero_grad()


def train(args, r, dset, model, tokenizer):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize training things
    model = model.to(device)
    wte = model._modules['transformer'].wte.to(device)
    dataloader = DataLoader(
        dset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # set up saving
    save_dir_unique = datetime.now().strftime("%b_%d_%H_%M_") + \
        ''.join(random.choices(string.ascii_lowercase, k=12))
    save_dir = os.path.join(args.save_dir, save_dir_unique)
    logging.info('saving to ' + save_dir)

    # initialize prefix
    if args.prefix_or_suffix == 'prefix':
        # get embedding of a chosen prefix string as nn.Parameter
        search_emb = search.get_init_prefix(
            model, dataloader, tokenizer, wte, device)

        # actually do fitting and saving
        train_prefix(args, r, model, wte, dataloader,
                     device, save_dir, search_emb)

    elif args.prefix_or_suffix == 'suffix':
        train_suffix(args, r, model, dataloader, device, save_dir)

if __name__ == '__main__':

    # initialize args
    def init_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=300,
                            help='batch size for training')
        parser.add_argument('--seed', type=int, default=1,
                            help='random seed')
        parser.add_argument('--n_epochs', type=int, default=10000,
                            help='number of epochs for training')
        parser.add_argument('--max_digit', type=int, default=100,
                            help='maximum value of each digit in summand')
        parser.add_argument('--save_dir', type=str, default='results',
                            help='directory for saving')
        parser.add_argument('--epoch_save_interval', type=int, default=1,
                            help='interval to save results')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='learning rate')
        parser.add_argument('--checkpoint', type=str, default="EleutherAI/gpt-neo-2.7B",
                            help='model checkpoint to use')
        parser.add_argument('--prefix_or_suffix', type=str, default="prefix",  # either prefix or suffix
                            help='model checkpoint to use')
        return parser
    parser = init_parser()
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # load model and data
    logger.info('loading model and data...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    dset = data.get_data(max_digit=args.max_digit)

    # set up saving
    r = defaultdict(list)
    r.update(vars(args))

    # train
    logger.info('beginning training...')
    train(args, r, dset, model, tokenizer)
