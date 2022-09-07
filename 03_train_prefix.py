from typing import Dict, List
import dataclasses
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
    PrefixTunedModel, PromptTunedModel, HotFlipPrefixTunedModel, GumbelPrefixTunedModel
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
    'gumbel': GumbelPrefixTunedModel,
    'hotflip': HotFlipPrefixTunedModel,
    'prompt_tune': PromptTunedModel,
}


@dataclasses.dataclass
class PrefixLoss:
    """Computes next-token-prediction loss with optional language modeling component.
    """
    gamma: float

    def _compute_fluency_loss(
            self, logits: torch.Tensor, input_ids: torch.Tensor
        ) -> torch.Tensor:
        if self.gamma == 0:
            return torch.tensor(0.0).to(device)
        log_probs = logits.log_softmax(dim=-1)
        num_input_words = input_ids.shape[1]
        log_probs_for_input = log_probs[:, -1-num_input_words:-1, :]
        input_log_probs = torch.gather(
            log_probs_for_input, dim=2, index=input_ids[...,None].to(device)
        )
        return -1 * input_log_probs.mean()

    def _compute_token_loss(
            self, next_token_logits: torch.Tensor, next_token_idxs: torch.Tensor, answer_mask: torch.Tensor
        ) -> torch.Tensor:
        batch_size, vocab_size = next_token_logits.shape
        assert next_token_idxs.shape == (batch_size,)
        assert answer_mask.shape == (vocab_size,)
        next_token_logits = torch.where(
            answer_mask[None], next_token_logits, torch.tensor(float('-inf')).to(device)
        )
        return torch.nn.functional.cross_entropy(
            input=next_token_logits,
            target=next_token_idxs,
            reduction='mean'
        )
    
    def __call__(self,
            input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            logits: torch.Tensor,
            answer_mask: torch.Tensor,
        ) -> torch.Tensor:
        """Computes loss.

        Args:
            input_ids (int torch.Tensor): array of token IDs for inputs
            next_token_ids (int torch.Tensor): array of token IDs for the word
                that comes after the input
            logits (float torch.Tensor): logits for all output tokens, including
                the next one
            answer_mask (bool torch.Tensor): mask over tokens to remove irrelevant ones

        Returns: float torch.Tensor scalar, loss value (lower is better).
        """
        fluency_loss = (
            self._compute_fluency_loss(
                logits=logits,
                input_ids=input_ids
            )
        )
        token_loss = (
            self._compute_token_loss(
                next_token_logits=logits[:, -1, :],
                next_token_idxs=next_token_ids,
                answer_mask=answer_mask,
            )
        )
        return token_loss + (self.gamma * fluency_loss)


def train(
        args: argparse.Namespace,
        r: Dict[str, List],
        dset: datasets.Dataset,
        model: PrefixTunedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        gamma: float
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

    loss_fct = PrefixLoss(gamma=gamma)
    
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
            # todo; update template
            x_text = [prompt.replace('Given ', '') for prompt in batch['input']]
            y_text = [answer.replace('.', '').rstrip() for answer in batch['output']] # strip newlines and periods.
            full_text = [x_text[i] for i in range(len(x_text))]

            # calculate loss
            # currently this calculates loss only on the answer token
            idxs_correct = tokenizer(y_text, return_tensors='pt')['input_ids'].to(device)
            try:
                assert idxs_correct.nelement() == len(y_text), 'For now assume that each answer is a single token'
            except:
                print("error!")
                breakpoint()
            idxs_correct = idxs_correct.squeeze(dim=1)

            input_ids, outputs = model.forward_text(text=full_text)

            total_n += len(idxs_correct)
            total_n_correct += (outputs['logits'][:, -1, :].argmax(dim=-1) == idxs_correct).int().sum()

            loss = loss_fct(input_ids=input_ids, next_token_ids=idxs_correct, logits=outputs['logits'], answer_mask=possible_answer_mask)

            all_losses.append(loss)
            loss.backward()
            pbar.set_description(f"Loss = {torch.tensor(all_losses).mean():.3f}")

            # optimize
            optim.step()
            optim.zero_grad()
        
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch}. average loss = {avg_loss:.3f} / {total_n_correct} / {total_n} correct ({total_n_correct/total_n*100:.2f}%)")

        # save stuff
        for key, val in model.compute_metrics().items():
            r[key].append(val)

        r['losses'].append(avg_loss)
        if epoch % args.epoch_save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

        model.post_epoch()

        # optimize
        # optim.step()
        # optim.zero_grad()


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
                            "EleutherAI/gpt-neo-125M",
                            "EleutherAI/gpt-neo-1.3B",
                            "EleutherAI/gpt-neo-2.7B",
                            "gpt2",        # 117M params
                            "gpt2-medium", # 355M params
                            "gpt2-large",  # 774M params
                            "gpt2-xl",     # 1.5B params
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
    model = model_cls_dict[args.model_cls](model=lm, tokenizer=tokenizer)
    dset, check_answer_func, description = data.get_data(args=args, task_name=args.task_name)
    print(f"got task with description: {description}")

    logger.info('beginning training...')
    r = train(args=args, r=r, dset=dset, model=model, tokenizer=tokenizer, gamma=args.gamma)
