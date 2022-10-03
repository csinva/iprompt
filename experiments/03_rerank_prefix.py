from typing import Callable, Dict, List

from collections import defaultdict
from copy import deepcopy
import logging
import os
import random
import string
import argparse

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import iprompt.data_utils.data as data
from model_utils.prefix import get_prefix_from_mlm, compute_log_ppl_loss
import iprompt.utils as utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_best_prompts_mlm(
        args: argparse.Namespace,
        r: Dict[str, List],
        dset: datasets.Dataset,
        lm_checkpoint: str,
        check_answer_func: Callable[[str], bool],
        mlm_name: str,
        mlm_num_candidates: int,
        do_reranking: bool
    ):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """
    suffix_str = data.get_init_suffix(args)
    prefix_template = suffix_str + '{mask}.'

    dataloader = DataLoader(
        dset, 
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    logger.info('computing prefixes with model %s', mlm_name)
    prefix_list = get_prefix_from_mlm(
        dataloader=dataloader,
        mlm_name=mlm_name,
        num_candidates=mlm_num_candidates,
        template=prefix_template
    )
    r["prefixes"] = prefix_list
    r["prefixes__check_answer_func"] = list(map(check_answer_func, prefix_list))

    if do_reranking:
        return rerank_prefix_list(
            args=args,
            r=r,
            dataloader=dataloader,
            lm_checkpoint=lm_checkpoint,
            prefix_list=prefix_list
        )
    else:
        return r

def rerank_prefix_list(
        args: argparse.Namespace,
        r: Dict[str, List],
        dataloader: DataLoader,
        lm_checkpoint: str,
        prefix_list: List[str]
    ):
    """
    Params
    ------
    r: dict
        dictionary of things to save
    """

    logger.info('got %d prefixes, now computing losses', len(prefix_list))


    if args.llm_float16:
        lm = AutoModelForCausalLM.from_pretrained(
            checkpoint, output_hidden_states=False,
            revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:    
        lm = AutoModelForCausalLM.from_pretrained(
            lm_checkpoint, output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained(lm_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

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
                list(map(process_s, batch['output'])),
                return_tensors='pt', add_special_tokens=False,
                truncation=True, max_length=1
            )['input_ids'].to(device).flatten()
            input_ids = torch.cat(
                (
                    # add BOS token in case we want to compute full-input fluency
                    torch.tensor([tokenizer.bos_token_id]).repeat((this_batch_size, 1)).to(device),
                    prefix_tokenized.input_ids.repeat((this_batch_size, 1)),
                    tokenized_text.input_ids,
                ),
                dim=1
            ).to(device)
            attention_mask = torch.cat(
                (
                    # extra attention for adding BOS token
                    torch.tensor([1]).repeat((this_batch_size, 1)).to(device),
                    prefix_tokenized.attention_mask.repeat((this_batch_size, 1)),
                    tokenized_text.attention_mask,
                ),
                dim=1
            ).to(device)

            with torch.no_grad():
                outputs = lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            assert outputs.logits.shape == input_ids.shape + (len(tokenizer.vocab),)

            # actually compute the loss.
            ###########################################################################
            # get index of ID token
            # we have to do this by comparing input to output. We have to flip to get the
            # *last* argmax, in case the output is repeated from the input (for example,
            # take "Input: Mexico Output: Mexico City" or even worse "Input: San Marino...")
            input_tokens_equal_to_output_tokens = (input_ids == tokenized_output_ids[:, None])
            sequence_length = input_tokens_equal_to_output_tokens.shape[1]
            output_token_ids = (
                sequence_length - 1 
                                - input_tokens_equal_to_output_tokens.flip(dims=[1]).int().argmax(dim=1)
            )
            ###########################################################################
            # next-token-only (few-shot) loss.
            next_token_logits = outputs.logits[torch.arange(this_batch_size), output_token_ids-1]
            total_loss += torch.nn.functional.cross_entropy(
                input=next_token_logits, target=tokenized_output_ids
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
                (next_token_logits.argmax(dim=1) == tokenized_output_ids).float().sum()
            )
        r["losses"].append((total_loss / total_n).item())
        r["accs"].append((total_n_correct / total_n).item())
        print(f"Loss = {(total_loss / total_n):.4f} / Acc = {(total_n_correct/total_n):.2f} / Prefix '{prefix}'")
    
    #
    # breakpoint()
    # df = pd.DataFrame.from_dict({k:v for k,v in r.items() if k != 'task_name_list'})
    # print(df.sort_values(by='accs', ascending=False).head(n=20)[['prefixes', 'accs']])
    #

    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size for training')
    parser.add_argument('--max_dset_size', type=int,
                        default=10000, help='maximum allowable dataset size')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='number of epochs for training')
    parser.add_argument('--max_digit', type=int, default=100,
                        help='maximum value of each digit in summand')
    parser.add_argument('--template_num_init_string', type=int, default=0,
                        help='the number of the manually-specified prefix to be initialize with')
    parser.add_argument('--template_num_task_phrasing', type=int, default=0,
                        help='the number of the manual template for any given task (number of options varies with task')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')
    parser.add_argument('--epoch_save_interval', type=int, default=1,
                        help='interval to save results')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--mlm_num_candidates', type=int, default=128,
        help='number of candidates for single-instance text infilling')
    parser.add_argument('--mlm_name', type=str, default='roberta-large',
        help='model to use for MLM-based text infilling')
    parser.add_argument('--do_reranking', type=int, default=1,
                        help='boolean 0 or 1: whether to do re-ranking using a LLM')
    parser.add_argument('--task_name', type=str, default='add_two',
                        choices=(data.TASKS.keys() - {'SUFFIX'}),
                        help='name of task')
    parser.add_argument('--task_name_list', nargs="*", default=None,
                        help='names of tasks as list; alternative to passing task_name')
    parser.add_argument('--n_shots', type=int, default=1,
                        help='number of shots in the prompt')
    parser.add_argument('--max_num_samples', type=int, default=0,
                        help='if true, ranks prompts based on just a certain number of data samples')
    parser.add_argument('--use_cache', type=int, default=1,
                        help='boolean 0 or 1: whether to check for cache')
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
                        help='model checkpoint to use'
    )
    parser.add_argument('--llm_float16', '--float16', '--parsimonious', type=int, default=0, choices=(0, 1),
                        help='if true, loads LLM in fp16 and at low-ram')
    args = parser.parse_args()
    args_ignore_for_caching = {k for k in vars(
        args) if not k in vars(parser.parse_args([])).keys()}

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # iterate over tasks
    if args.task_name_list is not None:
        logging.info('using task_name_list ' + str(args.task_name_list))
    else:
        args.task_name_list = [args.task_name]
    for task_idx, task_name in enumerate(args.task_name_list):
        print(f'*** Executing task {task_idx+1}/{len(args.task_name_list)}')
        # actually set the task
        args.task_name = task_name
        
        # set up saving directory before seeding
        save_dir_unique_hash = utils.get_unique_dir_hash(parser, args, args_ignore_for_caching)
        save_dir_random_suffix = ''.join(
            random.choices(string.ascii_lowercase, k=4))
        save_dir = os.path.join(
            args.save_dir, save_dir_unique_hash + save_dir_random_suffix)
        logging.info(f'\n\nrunning {task_name} + saving to ' + save_dir)

        # check for cached run with these same args
        if args.use_cache and utils.check_cached(save_dir_unique_hash, args, args_ignore_for_caching, parser, args.save_dir):
            logging.info(f'cached version exists for {task_name}!\nsuccessfully skipping :)\n\n\n')
            continue

        logger.info('loading model and data...')
        checkpoint = args.checkpoint
        dset, check_answer_func, description = data.get_data(
            args=args, task_name=args.task_name, n_shots=args.n_shots, max_dset_size=args.max_dset_size,
        )

        print(f'Attempting task with description: "{description}"')

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        transformers.set_seed(args.seed)

        # Support computing things from just a single ´xample
        dset = dset.shuffle()
        if (args.max_num_samples > 0):
            dset = dset.filter(lambda _, i: (i < args.max_num_samples), with_indices=True)

        logger.info('beginning training...')

        r = defaultdict(list)
        r.update(vars(args))
        utils.save_json(args=args, save_dir=save_dir, fname='params.json', r=r)    
        r = find_best_prompts_mlm(
            args=args,
            r=r, dset=dset,
            lm_checkpoint=checkpoint,
            check_answer_func=check_answer_func,
            mlm_name=args.mlm_name, mlm_num_candidates=args.mlm_num_candidates,
            do_reranking=bool(args.do_reranking)
        )
        utils.save_json(args=args, save_dir=save_dir, fname='results.json', r=r)

