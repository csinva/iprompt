from typing import Dict, List

import itertools
import os
from os.path import dirname
from os.path import join as oj
import sys
import tempfile
import random


repo_dir = dirname(dirname(os.path.abspath(__file__)))
# SAVE_DIR = '/home/chansingh/mntv1/'
SAVE_DIR = f'/home/chansingh/mntv1/iprompt_revision_xmas/'
NUM_LEARNED_TOKENS = [6]
SEEDS = [1, 2, 3]
JOB_SUFFIX = 'long_suffs'
iprompt_criterion = ['loss'] # 'loss', 'acc'
# iprompt_criterion = ['acc'] # 'loss', 'acc'
PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size', 'float16'): [
        # ('gpt2', 32, 0),
        # ('gpt2-medium', 200, 0),
        # ('gpt2-large', 100, 0),
        # ('gpt2-xl', 32, 0),
        # ('EleutherAI/gpt-neo-2.7B', 16, 0),
        # ('EleutherAI/gpt-j-6B', 64, 1),
        # ('EleutherAI/gpt-j-6B', 16, 1),
        # ('EleutherAI/gpt-neox-20b', 1, 0),
        # ("facebook/galactica-6.7b", 64, 0), # which language model to use
        # ("facebook/galactica-6.7b", 64, 1), # which language model to use
        ('google/flan-t5-xl', 1, 0),
        # ('google/flan-t5-xxl', 1, 1)
    ],
}
NUM_LEARNED_TOKENS = [6]
SEEDS = [1, 2, 3]

def combine_param_dicts(PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT):
    # shared
    ks_shared = list(PARAMS_SHARED_DICT.keys())
    vals_shared = [PARAMS_SHARED_DICT[k] for k in ks_shared]
    for val in vals_shared:
        assert isinstance(
            val, list), f"param val {val} must be type list, got type {type(val)}"
    param_tuples_list_shared = list(
        itertools.product(*vals_shared))

    # coupled
    ks_coupled = list(PARAMS_COUPLED_DICT.keys())
    vals_coupled = [PARAMS_COUPLED_DICT[k] for k in ks_coupled]
    param_tuples_list_coupled = list(
        itertools.product(*vals_coupled))
    param_tuples_list_coupled_flattened = [
        sum(x, ()) for x in param_tuples_list_coupled]

    # final
    ks_final = ks_shared + list(sum(ks_coupled, ()))

    param_combos_final = [shared + combo
                          for shared in param_tuples_list_shared
                          for combo in param_tuples_list_coupled_flattened]
    return ks_final, param_combos_final


def run_command_bash(cmd: str) -> None:
    os.system(cmd)


def run_command_slurm(
    python_cmd: str,
    save_dir: str,
    gpu_str: str,
    mem_str: str = '32G',
    num_cpus: int = 1,
) -> None:
    dir_path = dirname(os.path.realpath(__file__))
    slurm_template_file = open(oj(dir_path, 'slurm_template.slurm'))
    new_slurm_file_text = slurm_template_file.read().format(
        save_dir=save_dir,
        job_name='interpretable-autoprompting',
        gpu=gpu_str,
        mem=mem_str,
        num_cpus=str(num_cpus),
        python_cmd=python_cmd
    )
    tmp_slurm_file = tempfile.NamedTemporaryFile(
        dir=save_dir, suffix=".tmp", delete=False
    )
    tmp_slurm_file.write((new_slurm_file_text + '\n').encode())
    tmp_slurm_file.close()
    # will block until slurm processes file.
    run_command_bash(f'sbatch {tmp_slurm_file.name}')
    print(
        f'launched slurm command, removing temporary file {tmp_slurm_file.name}')
    os.remove(tmp_slurm_file.name)


def run_dicts(
    ks_final: List,
    param_combos_final: List,
    cmd_python: str = 'python',
    script_name: str = '02_train_suffix.py',
    actually_run: bool = True,
    use_slurm: bool = False,
    shuffle: bool = False,
    reverse: bool = False,
    slurm_gpu_str: str = 'gpu:1',
    save_dir: str = '',
):
    if shuffle:
        random.shuffle(param_combos_final)
    if reverse:
        param_combos_final = param_combos_final[::-1]
    for i in range(len(param_combos_final)):
        param_str = cmd_python + ' ' + \
            os.path.join(repo_dir, script_name + ' ')
        for j, key in enumerate(ks_final):
            param_val = param_combos_final[i][j]
            if isinstance(param_val, list):
                param_str += '--' + key + ' ' + ' '.join(param_val) + ' '
            else:
                param_str += '--' + key + ' ' + str(param_val) + ' '
        print(
            f'\n\n-------------------{i + 1}/{len(param_combos_final)}--------------------\n', param_str)
        try:
            if actually_run:
                if use_slurm:
                    run_command_slurm(
                        python_cmd=param_str,
                        gpu_str=slurm_gpu_str,
                        save_dir=save_dir
                    )
                else:
                    run_command_bash(param_str)
        except Exception as e:
            print(e)
