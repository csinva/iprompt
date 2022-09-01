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
import logging
import pickle as pkl
from torch.utils.data import DataLoader
from datetime import datetime


def train(args, r, dset, model, tokenizer):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda'

    model.train() 

    model = model.to(device)
    trans = model._modules['transformer']
    wte = trans.wte.to(device)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    TOP_K = 50

    # track token-to-word mapping 
    rvocab = {v:k for k,v in tokenizer.vocab.items()}

    # set up saving
    save_dir_unique = datetime.now().strftime("%b_%d_%H_%M_") + ''.join(random.choices(string.ascii_lowercase, k=12))
    save_dir = os.path.join(args.save_dir, save_dir_unique)
    logging.info('saving to ' + save_dir)

    emb_dim = wte.weight.shape[1] # 768 or 2560 for some larger models

    # initialize prefix
    # prefix_str = [" interns"]
    # prefix_str = ['ャ']
    prefix_str = ['x']
    prefix_inputs = tokenizer(prefix_str, return_tensors="pt").to(device)
    prefix_emb = wte.forward(prefix_inputs['input_ids'])

    assert prefix_emb.shape == (1, 1, emb_dim)


    #####################################################################
    # trainable_prefix_emb = torch.nn.Parameter(wte.weight.mean(dim=0, keepdim=True)[None], requires_grad=True).to(device)
    # trainable_prefix_emb = torch.nn.Parameter(torch.randu((1, 1, emb_dim)), requires_grad=True).to(device)
    trainable_prefix_emb = torch.nn.Parameter(prefix_emb[:, 0, :], requires_grad=True).to(device)
    #####################################################################

    # this code assumes that 'x' is the first token

    # optimizer
    optim = torch.optim.Adam([trainable_prefix_emb], lr=args.lr)

    assert model.training
    for epoch in range(args.n_epochs):
        # Print closest tokens at the beginning of each epoch.
        print("*" * 30)
        print(f"Epoch {epoch}. Closest tokens to x:")
        word_distances =  ((wte.weight - trainable_prefix_emb.reshape(1, emb_dim))**2).sum(1)
        assert word_distances.shape == (50_257,)
        topk_closest_words = distances = word_distances.topk(k=TOP_K, largest=False)
        for _id, _dist in zip(topk_closest_words.indices.cpu().tolist(), topk_closest_words.values.cpu().tolist()):
            print(f'\t{rvocab[_id]} ({_id}): {_dist:.3f}')
        print("*" * 30)

        all_losses = []
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in pbar:
            x_text = batch['input']
            y_text = batch['output']
            full_text = [x_text[i] for i in range(len(x_text))]
            # print(full_text)
            # breakpoint()

            ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            # print(ex_inputs)
            ex_embs = wte.forward(ex_inputs['input_ids'].to(
                device)).to(device)

            # concatenate prefix + example
            emb = torch.cat(
                (
                    trainable_prefix_emb.repeat(ex_embs.shape[0], 1, 1),
                    ex_embs
                ), 
                dim=1
            )

            # go through model
            outputs = model(inputs_embeds=emb)

            breakpoint()

            # calculate loss
            # currently this calculates loss only on the answer token
            idxs_correct = tokenizer(y_text, return_tensors='pt')['input_ids'].to(device)
            try:
                assert idxs_correct.nelement() == args.batch_size, 'For now assume that answer is a single token'
            except:
                breakpoint()
            # (batch_size, seq_len, vocab_size)

            last_token_logprobs = outputs['logits'][:, -1, :].log_softmax(dim=-1)
            correct_token_logprobs = torch.gather(last_token_logprobs, 1, idxs_correct)

            # accumulate gradients in this batch
            loss = -1 * correct_token_logprobs.mean() # minimize prob answer being wrong
            all_losses.append(loss.item())
            loss.backward()
            pbar.set_description(f"Loss = {loss:.3f}")

            # breakpoint()
            # optimize
            # optim.step()
            # optim.zero_grad()
        
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch}. average loss = {avg_loss:.3f}")

        # save stuff
        r['embs'].append(trainable_prefix_emb.detach().cpu().numpy())
        r['grads'].append(trainable_prefix_emb.grad.detach().cpu().numpy())
        r['losses'].append(avg_loss)
        if epoch % args.epoch_save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))
        # print('losses', loss)

        token_grads = wte.weight.mv(trainable_prefix_emb.grad.flatten())
        print(f"Epoch {epoch}. Most negative grads for x:")
        assert token_grads.shape == (50_257, )
        topk_smallest_grads = distances = token_grads.topk(k=TOP_K, largest=False)
        for _id, _dist in zip(topk_smallest_grads.indices.cpu().tolist(), topk_smallest_grads.values.cpu().tolist()):
            print(f'\t{rvocab[_id]} ({_id}): {_dist:.3f}')
        print("*" * 30)
        print(f"Epoch {epoch}. Most positive grads for x:")
        topk_largest_grads = distances = token_grads.topk(k=TOP_K, largest=True)
        for _id, _dist in zip(topk_largest_grads.indices.cpu().tolist()[::-1], topk_largest_grads.values.cpu().tolist()[::-1]):
            print(f'\t{rvocab[_id]} ({_id}): {_dist:.3f}')
        print("*" * 30)
        breakpoint()

        # optimize
        # optim.step()
        # optim.zero_grad()

        if epoch % 5 == 0:
            breakpoint()




    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=500,
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
    args = parser.parse_args()
    r = defaultdict(list)
    r.update(vars(args))
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    logger.info('loading model and data...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    dset = data.get_data(max_digit=args.max_digit, template_idx=0)

    logger.info('beginning training...')
    r = train(args, r, dset, model, tokenizer)
