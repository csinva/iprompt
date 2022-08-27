import os
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
import logging
import pickle as pkl
from torch.utils.data import DataLoader


def train(args, r, dset, model, tokenizer):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    np.random.seed(13)
    torch.manual_seed(13)
    device = 'cuda'

    model = model.to(device)
    trans = model._modules['transformer']
    wte = trans.wte.to(device)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

    # initialize prefix
    prefix_str = ["x the following two numbers: "]
    prefix_inputs = tokenizer(prefix_str, return_tensors="pt").to(device)
    prefix_emb = wte.forward(prefix_inputs['input_ids'])
    prefix_emb = torch.nn.Parameter(prefix_emb).to(device)

    # optimizer
    optim = torch.optim.Adam([prefix_emb], lr=args.lr)
    for epoch in range(args.n_epochs):
        for batch in tqdm(dataloader):
            x_text = batch['input']
            y_text = batch['output']
            full_text = [x_text[i] + y_text[i] for i in range(len(x_text))]
            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            ex_embs = wte.forward(ex_inputs['input_ids'].to(
                device)).to(device)

            # concatenate prefix + example
            emb = torch.cat((prefix_emb.repeat(args.batch_size, 1, 1),
                            ex_embs), dim=1)

            # go through model
            outputs = model(inputs_embeds=emb)

            # calculate loss
            idxs_correct = tokenizer(y_text, return_tensors='pt')['input_ids']
            assert idxs_correct.nelement() == args.batch_size, 'For now assume that answer is a single token'
            y_idx_correct = idxs_correct[0]
            # (batch_size, seq_len, vocab_size)
            logit_answer = outputs['logits'][0, -1, y_idx_correct]

            # optimize
            optim.zero_grad()
            loss = -1 * logit_answer
            loss.backward()
            optim.step()

        # save stuff
        r['embs'].append(prefix_emb.detach().cpu().numpy())
        r['losses'].append(loss.item())
        # print('losses', loss)

        if epoch % args.epoch_save_interval == 0:
            pkl.dump(r, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs for training')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')
    parser.add_argument('--epoch_save_interval', type=int, default=1,
                        help='interval to save results')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    args = parser.parse_args()
    r = defaultdict(list)
    r.update(vars(args))

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    logger.info('loading model and data...')
    checkpoint = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    dset = data.get_data(N=10000)

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info('beginning training...')
    r = train(args, r, dset, model, tokenizer)
