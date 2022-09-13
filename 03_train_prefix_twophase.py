from typing import Dict, List

import logging
import os
import random
import string
import torch
import argparse
from copy import deepcopy
from collections import defaultdict

import datasets
from datasets import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import data
from model_utils.prefix import get_prefix_from_mlm, compute_log_ppl_loss
import utils



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
        args: argparse.Namespace,
        r: Dict[str, List],
        dset: datasets.Dataset,
        lm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        mlm_name: str,
        mlm_num_candidates,
    ):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    logger.info('computing prefixes with model %s', mlm_name)
    prefix_list = get_prefix_from_mlm(dataloader=dataloader, mlm_name=mlm_name, num_candidates=mlm_num_candidates)

    logger.info('got %d prefixes, now computing losses', len(prefix_list))

    lm.eval() 
    lm.to(device)
    
    vocab_size = len(tokenizer.vocab)

    for prefix in tqdm(prefix_list, desc='testing prefixes'):
        prefix_tokenized = tokenizer(prefix, return_tensors='pt', add_special_tokens=False).to(device)
        total_n = 0
        total_n_correct = 0.0
        total_loss = 0.0
        # Strip punctuation and spaces so we only compute loss
        # across the output *token*.
        process_s = lambda s: s.rstrip().rstrip('.')
        for idx, batch in enumerate(dataloader):
            this_batch_size = len(batch['text'])
            # Remove spacing and punctuation that might be in initial phrasing.
            input_text = list(map(process_s, batch['text']))
            # Add a space between the prompt and the input-output pairs.
            input_text = [' ' + t for t in input_text]
            tokenized_text = tokenizer(
                input_text, return_tensors='pt', add_special_tokens=False,
                padding='longest'
            ).to(device)

            tokenized_output_ids = tokenizer(
                list(map(process_s, batch['output'])), return_tensors='pt', add_special_tokens=False
            )['input_ids'].to(device).flatten()
            input_ids = torch.cat(
                (
                    # add BOS token in case we want to compute full-input fluency
                    torch.tensor([tokenizer.bos_token_id]).repeat((this_batch_size, 1)).to(device),
                    prefix_tokenized.input_ids.repeat((this_batch_size, 1)),
                    tokenized_text.input_ids,
                ),
                dim=1
            )
            attention_mask = torch.cat(
                (
                    # extra attention for adding BOS token
                    torch.tensor([1]).repeat((this_batch_size, 1)).to(device),
                    prefix_tokenized.attention_mask.repeat((this_batch_size, 1)),
                    tokenized_text.attention_mask,
                ),
                dim=1
            )

            with torch.no_grad():
                outputs = lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            assert outputs.logits.shape == input_ids.shape + (len(tokenizer.vocab),)

            # actually compute the loss.
            ###########################################################################
            # next-token-only (few-shot) loss.
            breakpoint()
            last_token_logits = outputs.logits[:, -2, :]
            total_loss += torch.nn.functional.cross_entropy(
                input=last_token_logits, target=tokenized_output_ids
            )
            ###########################################################################
            # Trim off prefix logits so we only compute loss on the input.
            # TODO consider getting IDs of just "output" tokens in the input,
            # to do it that way.
            # prefix_length = prefix_tokenized['input_ids'].numel()
            # logits = outputs.logits[:, prefix_length:, :]
            # input_ids = input_ids[:, prefix_length:]
            # total_loss += compute_log_ppl_loss(logits=logits, input_ids=input_ids)
            ###########################################################################

            total_n += this_batch_size
            total_n_correct += (
                (last_token_logits.argmax(dim=1) == tokenized_output_ids).float().sum()
            )
        r["prefixes"].append(prefix)
        r["losses"].append((total_loss / total_n).item())
        r["accs"].append((total_n_correct / total_n).item())
        print(f"Loss = {(total_loss / total_n):.4f} / Acc = {(total_n_correct/total_n):.2f} / Prefix '{prefix}'")
    
    #
    # df = pd.DataFrame.from_dict(r)
    # breakpoint()
    #

    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--mlm_num_candidates', type=int, default=128,
        help='number of candidates for single-instance text infilling'
    )
    parser.add_argument('--mlm_name', type=str, default='roberta-large',
        help='model to use for MLM-based text infilling'
    )
    parser.add_argument('--task_name', type=str, default='add_two',
                        choices=(data.TASKS.keys() - {'SUFFIX'}),
                        help='name of task')
    parser.add_argument('--n_shots', type=int, default=1,
                        help='number of shots in the prompt')
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

    # set up saving directory before seeding, etc.
    save_dir_unique_hash = utils.get_unique_dir_hash(parser, args)
    save_dir_random_suffix = ''.join(
        random.choices(string.ascii_lowercase, k=4))
    save_dir = os.path.join(
        args.save_dir, save_dir_unique_hash + save_dir_random_suffix)
    logging.info('saving to ' + save_dir)
    print("*** save_dir =", save_dir)

    logger.info('loading model and data...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    dset, check_answer_func, description = data.get_data(
        args=args, task_name=args.task_name, n_shots=args.n_shots,
    )

    print(f'Attempting task with description: "{description}"')

    logger.info('beginning training...')
    r = train(args=args, r=r, dset=dset, lm=lm, tokenizer=tokenizer,
        mlm_name=args.mlm_name, mlm_num_candidates=args.mlm_num_candidates
    )
    utils.save_json(args=args, save_dir=save_dir, fname='results.json', r=r)

