from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pickle as pkl
import random
import os
from os.path import join as oj
from iprompt.data_utils import neuro
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from iprompt import suffix
import torch
import dvu
dvu.set_style()


def make_prompt_from_word_list(word_list):
    s = 'The following list of words are all part of the same semantic category: '
    s += ', '.join(word_list)
    s += '.\nThe semantic category they all belong to, in one word, is'
    return s


# initial set up
# checkpoint = 'EleutherAI/gpt-j-6B'
checkpoint = 'EleutherAI/gpt-neox-20b'
# checkpoint = 'gpt2-medium'
# save_dir = f'/home/chansingh/mntv1/fmri/{checkpoint.replace("/", "___")}/logits'
save_dir = f'/home/chansingh/results_fast/fmri/{checkpoint.replace("/", "___")}/logits'
device = 'cuda'
batch_size = 1  # make sure this divides into word_lists.shape[0]
shuffle = True
seeds = range(1, 11)
use_fp16 = True

# set up model
print('loading the model...')
if use_fp16:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=False, torch_dtype=torch.float16).to(device)
    model = model.half()
else:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=False)
print('model dtype', model.dtype)
model = model.to(device)

with torch.no_grad():
    # run one example
    word_lists = neuro.fetch_data(n_words=15)
    s = make_prompt_from_word_list(word_lists[0])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    ex_inputs = tokenizer([s], padding='longest',
                          return_tensors='pt').to(device)
    next_token_logits = suffix.get_next_token_logits(
        ex_inputs, model).squeeze().detach().cpu().numpy()

    # iterate and store over all examples
    os.makedirs(save_dir, exist_ok=True)
    vocab_size = next_token_logits.shape[-1]

    for seed in seeds:
        print('running seed', seed)

        # get the data and shuffle
        word_lists = neuro.fetch_data(n_words=15)
        random.seed(seed)
        for i in range(word_lists.shape[0]):
            random.shuffle(word_lists[i])

        # initialize logits
        all_logits = np.zeros((word_lists.shape[0], vocab_size))

        # batch_size is step
        for i in tqdm(range(0, word_lists.shape[0], batch_size)):

            # compute logits
            s = [make_prompt_from_word_list(wlist)
                 for wlist in word_lists[i: i + batch_size]]
            ex_inputs = tokenizer(s, padding='longest',
                                  return_tensors='pt').to(device)
            next_token_logits = suffix.get_next_token_logits(
                ex_inputs, model).detach().cpu().numpy()
            all_logits[i: i + batch_size] = next_token_logits

            # save
            # if i % 500 == 99:
                # pkl.dump(all_logits, open(
                    # oj(save_dir, f'all_logits_seed={seed}_{i + 1}.pkl'), 'wb'))
        pkl.dump(all_logits, open(
            oj(save_dir, f'all_logits_seed={seed}.pkl'), 'wb'))
