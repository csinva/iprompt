import os
import random
import string
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, top_k_top_p_filtering
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


def train_prefix(args, r, model, wte, dataloader, device, save_dir, prefix_emb):
    """Gradient-based optimization of the prefix
    """
    # optimizer
    optim = torch.optim.Adam([prefix_emb], lr=args.lr)

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
        utils.save(epoch, args, save_dir, r)
        # print('losses', loss)

        # optimize
        optim.step()
        optim.zero_grad()


def train_suffix(args, r, model, dataloader, device, suffix_str: str, save_dir,
                 num_tokens_to_add=8,
                 disallow_whitespace_tokens=True,
                 beam_size=1):
    """Here we find the suffix which maximizes the likelihood over all examples.
    The algorithms is basically to do beam search on the average prob distrs. over all examples.
    """

    # set up beam search
    # suffix_candidates = [suffix_str]

    # run training loop
    logging.info(f'num batches: {len(dataloader)} batch_size {args.batch_size}')
    for epoch in range(args.n_epochs):
        num_examples = 0
        cum_logits = None
        for idx, batch in tqdm(enumerate(dataloader)):

            # set up inputs
            text = batch['text']
            full_text = [text[i] + suffix_str
                         for i in range(len(text))]
            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            input_ids = ex_inputs['input_ids']

            # go through model
            outputs = model(input_ids)
            logits = outputs['logits']  # (batch_size, seq_len, vocab_size)
            # get logits of last hidden state, sum over batch_size
            next_token_logits = logits[:, -1, :].sum(axis=0)

            # accumulate logits
            if cum_logits is None:
                cum_logits = next_token_logits.detach()
            else:
                cum_logits += next_token_logits.detach()
            num_examples += len(text)

        # use averaged logits
        # keep only top k tokens with highest probability
        # keep the top tokens with cumulative probability >= top_p
        avg_logits = cum_logits / num_examples
        # print('shapes', logits.shape, next_token_logits.shape, cum_logits.shape)
        avg_logits = avg_logits.detach().cpu().numpy().squeeze()
        avg_probs = np.exp(avg_logits) # softmax
        avg_probs /= np.sum(avg_probs)
        k = 50
        # could also check out top_k_top_p_filtering (https://huggingface.co/docs/transformers/v4.16.2/en/task_summary)
        top_k_inds = np.argpartition(avg_logits, -k)[-k:] # get topk
        top_k_inds = top_k_inds[np.argsort(avg_logits[top_k_inds])][::-1] # sort the topk (largest first)        

        # decode and log
        top_decoded_tokens = np.array([tokenizer.decode(ind) for ind in top_k_inds])
        logging.info(str(epoch) + ' ' + repr(suffix_str))
        for i in range(20):
            logging.info('\t ' + repr(top_decoded_tokens[i]) + '\t' + f'{avg_probs[top_k_inds[i]]:.2E}')

        if disallow_whitespace_tokens:
            disallowed_idxs = np.array([s.isspace() for s in top_decoded_tokens], dtype=bool)
            top_k_inds = top_k_inds[~disallowed_idxs]
            top_decoded_tokens = top_decoded_tokens[~disallowed_idxs]

        suffix_str += top_decoded_tokens[0]
        r['beam_search'].append({
            suffix_str: {
                top_decoded_tokens[i]: avg_probs[top_k_inds[i]]
                for i in range(top_k_inds.shape[0])
            }
        })
        
        # save stuff
        utils.save(epoch, args, save_dir, r)


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
    dataloader = DataLoader(
        dset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # set up saving
    save_dir_unique = datetime.now().strftime("%b_%d_%H_%M_") + \
        ''.join(random.choices(string.ascii_lowercase, k=12))
    save_dir = os.path.join(args.save_dir, save_dir_unique)
    logging.info('saving to ' + save_dir)

    # initialize prefix
    if args.prefix_or_suffix.startswith('pre'):
        # extract out embedding
        wte = model._modules['transformer'].wte.to(device)

        # get embedding of a chosen prefix string as nn.Parameter
        prefix_emb = search.get_init_prefix(
            model, dataloader, tokenizer, wte, device)

        # actually do fitting and saving
        with torch.no_grad():
            train_prefix(args, r, model, wte, dataloader,
                         device, save_dir, prefix_emb)

    elif args.prefix_or_suffix.startswith('suf'):
        suffix_str = search.get_init_suffix(
            model, dataloader, tokenizer, device)
        train_suffix(args, r, model, dataloader, device, suffix_str, save_dir)


if __name__ == '__main__':
    # python3 01_train_toy_ex.py --prefix_or_suffix suffix --batch_size 200 --checkpoint EleutherAI/gpt-neo-2.7B 
    # python3 01_train_toy_ex.py --prefix_or_suffix suffix --batch_size 1 --checkpoint EleutherAI/gpt-neox-20b
    # python3 01_train_toy_ex.py --prefix_or_suffix suffix --batch_size 50 --checkpoint EleutherAI/gpt-j-6B
    # python3 01_train_toy_ex.py --prefix_or_suffix suffix --batch_size 10 --checkpoint EleutherAI/gpt-j-6B --n_shots 3
    # python3 01_train_toy_ex.py --prefix_or_suffix suffix --batch_size 100 --checkpoint EleutherAI/gpt-neo-2.7B --n_shots 3



    # initialize args
    def init_parser():
        parser = argparse.ArgumentParser()

        # dataset args
        parser.add_argument('--max_digit', type=int, default=100,
                            help='maximum value of each digit in summand')
        parser.add_argument('--n_shots', type=int, default=1,
                            help='number of shots in the prompt')

        # algorithm args
        parser.add_argument('--checkpoint', type=str, default="EleutherAI/gpt-j-6B", # EleutherAI/gpt-neox-20b, "EleutherAI/gpt-neo-2.7B"
                            help='model checkpoint to use')
        parser.add_argument('--prefix_or_suffix', type=str, default="prefix",  # either prefix or suffix (pre or suf will suffice)
                            help='model checkpoint to use')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='learning rate')

        # training misc args
        parser.add_argument('--batch_size', type=int, default=100,
                            help='batch size for training')
        parser.add_argument('--seed', type=int, default=1,
                            help='random seed')
        parser.add_argument('--n_epochs', type=int, default=10000,
                            help='number of epochs for training')

        # logging/saving args
        parser.add_argument('--save_dir', type=str, default='results',
                            help='directory for saving')
        parser.add_argument('--epoch_save_interval', type=int, default=1,
                            help='interval to save results')
        
        
        return parser
    parser = init_parser()
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.info(str(vars(args)))

    # load model and data
    logger.info('loading model and data...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=args.prefix_or_suffix == 'prefix')
    dset = data.get_data(n_shots=args.n_shots, max_digit=args.max_digit)

    # set up saving
    r = defaultdict(list)
    r.update(vars(args))

    # train
    logger.info('beginning training...')
    train(args, r, dset, model, tokenizer)
