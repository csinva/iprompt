import itertools
import os
from os.path import dirname
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))


PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size'): [
        # ('gpt2-medium', 200),
        # ('gpt2-large', 100),
        # ('gpt2-xl', 40),
        # ('EleutherAI/gpt-j-6B', 10)
        ('EleutherAI/gpt-neox-20b', 1),
    ],
}

def combine_param_dicts(PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT):

    # shared
    ks_shared = list(PARAMS_SHARED_DICT.keys())
    vals_shared = [PARAMS_SHARED_DICT[k] for k in ks_shared]
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


def run_dicts(ks_final, param_combos_final, cmd_python='python',
              script_name='02_train_suffix.py', actually_run=True):
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
                os.system(param_str)
        except Exception as e:
            print(e)
