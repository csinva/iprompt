import logging
import pickle as pkl
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

import data
import parallel
import utils


def get_avg_probs_next_token(args, suffix_str: str, model, dataloader, tokenizer):
    """Get the average probs for the next token across the entire dataset
    """
    num_examples = 0
    cum_logits = None
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        # set up inputs
        text = batch['text']
        full_text = [text[i] + suffix_str
                     for i in range(len(text))]
        ex_inputs = tokenizer(
            full_text, padding='longest', return_tensors='pt')
        ex_inputs = parallel.inputs_to_device(args, ex_inputs)

        # go through model
        outputs = model(
            input_ids=ex_inputs['input_ids'], attention_mask=ex_inputs['attention_mask'])
        logits = outputs['logits']  # (batch_size, seq_len, vocab_size)
        # get logits of last hidden state, sum over batch_size
        next_token_logits = logits[:, -1,
                                   :].sum(axis=0).log_softmax(dim=-1)

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
    avg_probs = np.exp(avg_logits)  # softmax
    avg_probs /= np.sum(avg_probs)

    return avg_logits


def get_probs_single_query_next_token(args, suffix_str: str, model, dataloader, tokenizer):
    """Get the average probs for the next token across the entire dataset
    """
    # get a single input
    batch = next(iter(dataloader))
    text = batch['text']
    full_text = [text[0] + suffix_str]
    ex_inputs = tokenizer(
        full_text, padding='longest', return_tensors='pt')
    ex_inputs = parallel.inputs_to_device(args, ex_inputs)

    # go through model
    outputs = model(
        input_ids=ex_inputs['input_ids'], attention_mask=ex_inputs['attention_mask'])
    logits = outputs['logits']  # (batch_size, seq_len, vocab_size)
    next_token_logits = (
        logits[:, -1, :]
        .sum(axis=0)  # sum over batch_size
        .log_softmax(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )

    return next_token_logits


def train_suffix(args, r, model, dataloader, check_answer_func, tokenizer, save_dir,
                 disallow_whitespace_tokens=True,
                 beam_size_for_saving=15):
    """Here we find the suffix which maximizes the likelihood over all examples.
    The algorithm is basically to do breadth-first beam-search on the next-token prob distr. averaged over all examples
    """

    # set up BFS beam search
    suffix_str = data.get_init_suffix(args)

    suffixes = [{'s': suffix_str, 'num_tokens_added': 0,
                 'running_prob': 1, 'num_suffixes_checked': 0}]
    r['suffix_str_init'] = suffix_str
    r['len_suffix_str_init'] = len(suffix_str)
    num_model_queries = 0
    logging.info(
        f'num batches: {len(dataloader)} batch_size {args.batch_size}')

    while len(suffixes) > 0:
        suffix_dict = suffixes.pop()
        suffix_str = suffix_dict['s']
        num_suffixes_checked = suffix_dict['num_suffixes_checked']

        # get avg_probs
        if args.use_single_query:
            avg_probs = get_probs_single_query_next_token(
                args, suffix_str, model, dataloader, tokenizer)
        else:
            avg_probs = get_avg_probs_next_token(
                args, suffix_str, model, dataloader, tokenizer)
        num_model_queries += 1

        # could also check out top_k_top_p_filtering
        # (https://huggingface.co/docs/transformers/v4.16.2/en/task_summary)
        top_k_inds = np.argpartition(
            avg_probs, -beam_size_for_saving)[-beam_size_for_saving:]  # get topk
        # sort the topk (largest first)
        top_k_inds = top_k_inds[np.argsort(avg_probs[top_k_inds])][::-1]

        # decode and log
        top_decoded_tokens = np.array(
            [tokenizer.decode(ind) for ind in top_k_inds])
        logging.info(str(num_model_queries) + ' ' + repr(suffix_str))
        for i in range(top_k_inds.size):
            logging.info(
                '\t ' + repr(top_decoded_tokens[i]) + '\t' + f'{avg_probs[top_k_inds[i]]:.2E}')

        if disallow_whitespace_tokens:
            disallowed_idxs = np.array([s.isspace()
                                       for s in top_decoded_tokens], dtype=bool)
            top_k_inds = top_k_inds[~disallowed_idxs]
            top_decoded_tokens = top_decoded_tokens[~disallowed_idxs]

        # save results for suffix_str
        r['suffix_str_added'].append(suffix_str[r['len_suffix_str_init']:])
        # if we made it here, we did not find the answer
        r['num_model_queries'].append(num_model_queries)
        r['running_prob'].append(suffix_dict['running_prob'])
        if args.use_verbose_saving:
            r['correct'].append(False)
            r['suffix_str_full'].append(suffix_str)
            r['decoded_token'].append(top_decoded_tokens[0])
            r['top_decoded_tokens_dict'].append({
                top_decoded_tokens[i]: avg_probs[top_k_inds[i]]
                for i in range(top_k_inds.shape[0])
            })
        utils.save(args, save_dir, r, epoch=None, final=True)

        # check each beam
        if suffix_dict['num_tokens_added'] < args.max_num_tokens:
            # take this max just in case all tokens were somehow whitespace
            for beam_num in range(max(args.beam_width_suffix, top_k_inds.size)):
                suffix_new = suffix_str + top_decoded_tokens[beam_num]
                if check_answer_func(suffix_new):  # and args.early_stopping
                    r['final_answer_full'] = suffix_new
                    r['final_answer_added'] = suffix_new[r['len_suffix_str_init']:]
                    r['final_model_queries'] = num_model_queries
                    r['final_num_suffixes_checked'] = num_suffixes_checked + \
                        beam_num + 1
                    logging.info('successful early stopping!')
                    logging.info('\t' + repr(r['suffix_str_init']))
                    logging.info('\t' + repr(r['final_answer_added']))
                    logging.info(save_dir)
                    utils.save(args, save_dir, r, final=True)
                    utils.save_json(r={  # save some key outputs in readable form
                        k: r[k]
                        for k in r if isinstance(r[k], str) or isinstance(r[k], int)
                    }, save_dir=save_dir, fname='final.json')
                    exit(0)

                # for bfs insert at beginning (dfs would append at end)
                suffixes.insert(0, {
                    's': suffix_new,
                    'num_tokens_added': suffix_dict['num_tokens_added'] + 1,
                    'running_prob': suffix_dict['running_prob'] * avg_probs[top_k_inds[beam_num]],

                    # checked beam_width at current suffix + all suffixes before this one (assumes BFS-beam search)
                    # this is the total number of suffixes checked at the time when this will be opened above
                    'num_suffixes_checked': num_suffixes_checked + args.beam_width_suffix * (beam_num + 1)
                })
