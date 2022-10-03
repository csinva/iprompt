import argparse
from copy import deepcopy
import logging
import os
import random
import string
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

import iprompt.data as data
import iprompt.parallel as parallel
import iprompt.suffix as suffix
import iprompt.utils as utils

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
                        default=1000, help='maximum allowable dataset size')
    parser.add_argument('--max_digit', type=int, default=10,
                        help='maximum value of each digit in summand')
    parser.add_argument('--template_num_init_string', type=int, default=0,
                        help='the number of the manually-specified prefix to be initialize with')
    parser.add_argument('--template_num_task_phrasing', type=int, default=0,
                        help='the number of the manual template for any given task (number of options varies with task')
    parser.add_argument('--train_split_frac', type=float,
                        default=None, help='fraction for train-test split if desired')

    # algorithm args
    # gpt # "gpt2-medium" (355M), "gpt2-large" (774M), "gpt2-xl" (1.5B)
    # gpneo # "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"
    parser.add_argument('--checkpoint', type=str, default="gpt2",
                        help='model checkpoint to use')
    parser.add_argument('--max_num_tokens', type=int, default=1,
                        help='max length of sequence to find (num tokens)')
    parser.add_argument('--beam_size', type=int, default=4,
                        help='max width of beam in suffix search')
    parser.add_argument('--beam_size_extra', type=int, default=50,
                        help='extra width of beam to check at each iteration')
    parser.add_argument('--use_single_query', type=int, default=0,
                        help='boolean 0 or 1: use baseline model? only uses a single example to prompt rather than the entire dset')
    parser.add_argument('--use_stopwords', type=int, default=1,
                        help='boolean 0 or 1: whether to allow stopwords when searching for prompt')
    parser.add_argument('--use_early_stopping', type=int, default=1,
                        help='whether to stop searching once finding correct answer')
    parser.add_argument('--use_generic_query', type=int, default=0,
                        help='whether to use a generic query template instead of a task-specific one (harder)')
    parser.add_argument('--float16', type=int, default=0,
                        help='whether to use float16 / low cpu mem')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='tmp',
                        help='directory for saving')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache, but currently it checks all non-defaults)
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
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for training')
    parser.add_argument('--task_name_list', nargs="*", default=None,
                        help='names of tasks as list; alternative to passing task_name')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_main = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_main))
    args = parser.parse_args()
    args_ignore_for_caching = {k for k in vars(
        args) if not k in vars(parser_main.parse_args([])).keys()}

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))

    # iterate over tasks
    if args.task_name_list is not None:
        logging.info('using task_name_list ' + str(args.task_name_list))
    else:
        args.task_name_list = [args.task_name]
    for task_name in args.task_name_list:
        # actually set the task
        args.task_name = task_name

        # set up saving directory before seeding
        save_dir_unique_hash = utils.get_unique_dir_hash(
            parser, args, args_ignore_for_caching)
        save_dir_random_suffix = ''.join(
            random.choices(string.ascii_lowercase, k=4))
        save_dir = os.path.join(
            args.save_dir, save_dir_unique_hash + save_dir_random_suffix)
        logging.info(f'\n\nrunning {task_name} + saving to ' + save_dir)

        # check for cached run with these same args
        if args.use_cache and utils.check_cached(save_dir_unique_hash, args, args_ignore_for_caching, parser, args.save_dir):
            logging.info(
                f'cached version exists for {task_name}!\nsuccessfully skipping :)\n\n\n')
            continue

        # set seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # load data
        logger.info('loading data...')
        data_kwargs = dict(
            args=args, task_name=args.task_name, n_shots=args.n_shots, train_split_frac=args.train_split_frac, max_dset_size=args.max_dset_size,
        )
        if args.train_split_frac:
            (dset, dset_test), check_answer_func, descr = data.get_data(**data_kwargs)
        else:
            dset, check_answer_func, descr = data.get_data(**data_kwargs)
        dataloader = DataLoader(
            dset, batch_size=min(args.batch_size, len(dset)), shuffle=True, drop_last=True)
        logging.info(
            f'num_examples: {dset.shape[0]}, num batches: {len(dataloader)}')
        logging.info(dset[0])

        # load model
        if not 'model' in locals():
            logger.info('loading model...')
            checkpoint = args.checkpoint
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            tokenizer.pad_token = tokenizer.eos_token
            if args.float16:
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, output_hidden_states=False,
                    revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
                    )     
            else:
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
