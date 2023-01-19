import itertools
import os
from os.path import dirname
import sys
sys.path.append('..')
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/iprompt_revision4/anli/'
save_dir = submit_utils.SAVE_DIR

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to average over
     'seed': submit_utils.SEEDS,
     'iprompt_criterion': submit_utils.iprompt_criterion,

    # things to vary
    'use_preprefix': [1],
    'iprompt_preprefix_str': ["'Output yes if the input'"],
    'n_shots': [5],
    'task_name_list': [[
        f'd3_{i}' for i in range(54)
    ]],
    # 'model_cls': ['iprompt', 'autoprompt'],
    'model_cls': ['suffix'],
    # 'model_cls': ['autoprompt'],
    'num_learned_tokens': submit_utils.NUM_LEARNED_TOKENS,

    # stopping criteria
    'max_dset_size': [5000],
    'max_n_datapoints': [5000],
    'early_stopping_steps': [50],

    # fixed params
    'max_length': [128],
    'train_split_frac': [0.75],
    'single_shot_loss': [1],
    'iprompt_generation_repetition_penalty': [1.5],
}
PARAMS_SHARED_DICT['save_dir'] = [save_dir]
PARAMS_COUPLED_DICT = submit_utils.PARAMS_COUPLED_DICT

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

print('running job')
submit_utils.run_dicts(
    ks_final, param_combos_final, cmd_python=cmd_python,
    script_name='03_train_prefix.py', actually_run=True,
    shuffle=True,
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
