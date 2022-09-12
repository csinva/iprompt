import argparse
from copy import deepcopy
import logging
import os
import pickle as pkl
import random
import string
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

import data
import parallel
import model_utils.suffix as suffix
import utils

# initialize args


def add_main_args(parser):
    """Note: caching uses the non-default values from parser to name the saving directory.
    Changing the default arg an argument will break compatibility with previous cached runs.
    """
    # dataset args
    parser.add_argument('--task_name', type=str, default='add_two',
                        help='name of task')
    parser.add_argument('--n_shots', type=int, default=1,
                        help='number of shots in the prompt')
    parser.add_argument('--max_dset_size', type=int,
                        default=10000, help='maximum allowable dataset size')
    parser.add_argument('--max_digit', type=int, default=10,
                        help='maximum value of each digit in summand')
    parser.add_argument('--template_num_init_string', type=int, default=0,
                        help='the number of the manually-specified prefix to be initialize with')
    parser.add_argument('--template_num_task_phrasing', type=int, default=0,
                        help='the number of the manual template for any given task (number of options varies with task')

    # algorithm args
    # gpt # "gpt2-medium" (355M), "gpt2-large" (774M), "gpt2-xl" (1.5B)
    # gpneo # "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"
    parser.add_argument('--checkpoint', type=str, default="gpt2-medium",
                        help='model checkpoint to use')
    parser.add_argument('--max_num_tokens', type=int, default=4,
                        help='max length of sequence to find (num tokens)')
    parser.add_argument('--beam_width_suffix', type=int, default=4,
                        help='max width of beam in suffix search')
    parser.add_argument('--use_single_query', type=int, default=0,
                        help='boolean 0 or 1: use baseline model? only uses a single example to prompt rather than the entire dset')
    # parser.add_argument('--early_stopping', dest='early_stopping', default=True,
    #     help='whether to stop searching once finding correct answer - for suffix, this currently has to be true',
    #     action='store_true')

    # training misc args
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)
    """
    parser.add_argument('--use_cpu_only', type=int, default=0,
                        help='boolean 0 or 1: whether to force everything onto cpu')
    parser.add_argument('--use_parallelformers', type=int, default=1,
                        help='boolean 0 or 1: whether to try and use parallelformers')
    parser.add_argument('--use_cache', type=int, default=1,
                        help='boolean 0 or 1: whether to check for cache')
    parser.add_argument('--use_verbose_saving', type=int, default=0,
                        help='boolean 0 or 1: whether to save verbose things')
    parser.add_argument('--epoch_save_interval', type=int, default=1,
                        help='interval to save results')
    return parser


if __name__ == '__main__':
    # python 01_train.py --batch_size 200 --checkpoint EleutherAI/gpt-neo-2.7B
    # python 01_train.py --batch_size 1 --checkpoint EleutherAI/gpt-neox-20b
    # python 01_train.py --batch_size 50 --checkpoint EleutherAI/gpt-j-6B
    # python 01_train.py --batch_size 10 --checkpoint EleutherAI/gpt-j-6B --n_shots 3
    # python 01_train.py --batch_size 100 --checkpoint EleutherAI/gpt-neo-2.7B --n_shots 3
    # python 01_train.py --batch_size 10 --checkpoint EleutherAI/gpt-j-6B --n_shots 3 --max_digit 10
    # python 01_train.py --save_dir /home/chansingh/mntv1/test2

    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    parser = add_computational_args(parser)
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    logger.info(str(vars(args)))

    # set up saving dirctory before seeding
    save_dir_unique_hash = utils.get_unique_dir_hash(parser, args)
    save_dir_random_suffix = ''.join(
        random.choices(string.ascii_lowercase, k=4))
    save_dir = os.path.join(
        args.save_dir, save_dir_unique_hash + save_dir_random_suffix)
    logging.info('saving to ' + save_dir)

    # check for cached run with these same args
    if args.use_cache and utils.check_cached(save_dir_unique_hash, args, parser, args.save_dir):
        logging.info('cached version exists!\nsuccessfully exiting :)')
        exit(0)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    logger.info('loading data...')
    dset, check_answer_func, descr = data.get_data(
        args, args.task_name, n_shots=args.n_shots)
    dataloader = DataLoader(
        dset, batch_size=min(args.batch_size, len(dset)), shuffle=True, drop_last=True)
    logging.info(
        f'num_examples: {dset.shape[0]}, num batches: {len(dataloader)}')
    logging.info(dset[0])

    # load model
    logger.info('loading model...')
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=False)
    model = parallel.model_to_device(args, model)

    # set up saving
    r = defaultdict(list)
    r.update(vars(args))
    utils.save_json(args=args, save_dir=save_dir, fname='params.json', r=r)

    # train
    with torch.no_grad():
        suffix.train_suffix(args, r, model, dataloader,
                                  check_answer_func, tokenizer, save_dir)
