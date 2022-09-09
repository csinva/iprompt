from typing import Dict, List
import datasets
import os
import random
import string
import numpy as np
import torch
from torch import nn
import transformers
import matplotlib.pyplot as plt
import argparse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from model_utils.prefix import (
    PrefixLoss, PrefixModel,
    PromptTunedModel, HotFlipPrefixModel, GumbelPrefixModel
)
import pandas as pd
from datasets import Dataset
import data
import logging
import pickle as pkl
from torch.utils.data import DataLoader
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_cls_dict = {
    'gumbel': GumbelPrefixModel,
    'hotflip': HotFlipPrefixModel,
    'prompt_tune': PromptTunedModel,
}


def train(
        args: argparse.Namespace,
        r: Dict[str, List],
        dset: datasets.Dataset,
        model: PrefixModel,
        tokenizer: transformers.PreTrainedTokenizer
    ):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model.train() 

    model = model.to(device)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # set up saving
    save_dir_unique = datetime.now().strftime("%b_%d_%H_%M_") + ''.join(random.choices(string.ascii_lowercase, k=12))
    save_dir = os.path.join(args.save_dir, save_dir_unique)
    logging.info('saving to ' + save_dir)

    # optimizer
    optim = torch.optim.AdamW(model.trainable_params, lr=args.lr)

    assert model.training
    
    # Compute loss only over possible answers to make task easier
    possible_answer_ids = []
    for batch in dataloader:
        y_text = [answer for answer in batch['output']]
        y_tokenized = tokenizer(y_text, return_tensors='pt')
        true_next_token_ids = y_tokenized['input_ids'][:, 0] # only test on the single next token
        possible_answer_ids.extend(true_next_token_ids.tolist())
    
    possible_answer_ids = torch.tensor(possible_answer_ids)
    num_unique_answers = len(set(possible_answer_ids.tolist()))
    assert num_unique_answers > 0, "need multiple answers for multiple choice"
    random_acc = 1 / num_unique_answers * 100.0
    majority_count = (possible_answer_ids[:, None] == possible_answer_ids[None, :]).sum(dim=1).max()
    majority_acc = majority_count * 100.0 / len(possible_answer_ids)
    print(f"Training with {num_unique_answers} possible answers / random acc {random_acc:.1f}% / majority acc {majority_acc:.1f}%")
    
    vocab_size = len(tokenizer.vocab)
    possible_answer_mask = (
        torch.arange(start=0, end=vocab_size)[:, None]
        == 
        possible_answer_ids[None, :]
    ).any(dim=1).to(device)

    for epoch in range(args.n_epochs):
        model.pre_epoch()

        all_losses = []
        
        total_n = 0
        total_n_correct = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in pbar:
            x_text = [f' {prompt}' for prompt in batch['input']]
            y_text = [answer.replace('.', '').rstrip() for answer in batch['output']] # strip newlines and periods.

            input_ids = model.tokenizer(x_text, return_tensors='pt')['input_ids'].to(device)
            next_token_ids = tokenizer(y_text, return_tensors='pt')['input_ids'].to(device).squeeze(dim=1)
            assert next_token_ids.nelement() == len(y_text), 'For now assume that each answer is a single token'

            loss, n_correct = model.compute_loss(
                original_input_ids=input_ids,
                next_token_ids=next_token_ids,
                possible_answer_mask=possible_answer_mask
            )
            loss.backward()

            total_n += len(next_token_ids)
            total_n_correct += n_correct

            all_losses.append(loss)
            pbar.set_description(f"Loss = {torch.tensor(all_losses).mean():.3f}")
        
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch}. average loss = {avg_loss:.3f} / {total_n_correct} / {total_n} correct ({total_n_correct/total_n*100:.2f}%)")

        # save stuff
        for key, val in model.compute_metrics().items():
            r[key].append(val)

        r['losses'].append(avg_loss)
        if epoch % args.epoch_save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

        model.post_epoch(dataloader=dataloader, possible_answer_mask=possible_answer_mask)

        # optimize
        optim.step()
        optim.zero_grad()


    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_cls', type=str,
                    choices=model_cls_dict.keys(),
                    required=True,
                    help='model type to use for training')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size for training')
    parser.add_argument('--max_dset_size', type=int,
                        default=10000, help='maximum allowable dataset size')
    parser.add_argument('--template_num_task_phrasing', type=int, default=0,
                        help='the number of the manual template for any given task (number of options varies with task')
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
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='hparam: weight for language modeling loss')
    parser.add_argument('--task_name', type=str, default='add_two',
                        choices=(data.TASKS.keys() - {'SUFFIX'}),
                        help='name of task'
    )
    parser.add_argument('--checkpoint', type=str, default="EleutherAI/gpt-neo-2.7B",
                        choices=(
                            ############################
                            "EleutherAI/gpt-neo-125M",
                            "EleutherAI/gpt-neo-1.3B",
                            "EleutherAI/gpt-neo-2.7B",
                            ############################
                            "EleutherAI/gpt-j-6B",
                            ############################
                            "EleutherAI/gpt-neox-20b",
                            ############################
                            "gpt2",        # 117M params
                            "gpt2-medium", # 355M params
                            "gpt2-large",  # 774M params
                            "gpt2-xl",     # 1.5B params
                            ############################
                        ),
                        help='model checkpoint to use')
    args = parser.parse_args()
    r = defaultdict(list)
    r.update(vars(args))
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    logger.info('loading model and data...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    lm = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    loss_func = PrefixLoss(gamma=args.gamma, tokenizer=tokenizer)
    model = model_cls_dict[args.model_cls](loss_func=loss_func, model=lm, tokenizer=tokenizer)
    dset, check_answer_func, description = data.get_data(args=args, task_name=args.task_name)
    print(f"got task with description: {description}")

    logger.info('beginning training...')
    r = train(args=args, r=r, dset=dset, model=model, tokenizer=tokenizer)
