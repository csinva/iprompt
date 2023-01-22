import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/iprompt_revision2/math/'
# save_dir = f'/home/chansingh/mntv1/iprompt_revision2/anli/'
# save_dir = f'/home/chansingh/mntv1/iprompt_revision4/anli/'
save_dir = submit_utils.SAVE_DIR

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to average over
    'seed': submit_utils.SEEDS,
    'iprompt_criterion': submit_utils.iprompt_criterion,
    # 'seed': [1],

    # things to vary
    'n_shots': [5],

    'task_name_list': [[
        'add_two', 'multiply_two', 
        'subtract_two',
        'max_two', 'first_two',
        'square_one', 'double_one',
        'exp_one',  'fibonacci_one',
        'divide_two', 
    ]],
    # 'model_cls': ['iprompt'],
    'model_cls': ['autoprompt'],
    # 'model_cls': ['iprompt', 'autoprompt'],
    'num_learned_tokens': submit_utils.NUM_LEARNED_TOKENS,
    # 'model_cls': ['autoprompt'], #, 'autoprompt'],
    # 'num_learned_tokens': [6, 12],

    # stopping criteria
    'max_dset_size': [5000],
    'max_n_datapoints': [5000],
    'early_stopping_steps': [25],

    # fixed params
    'max_digit': [10],
    'train_split_frac': [0.75],
    'single_shot_loss': [1],
}
PARAMS_SHARED_DICT['save_dir'] = [save_dir]
PARAMS_COUPLED_DICT = submit_utils.PARAMS_COUPLED_DICT

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

print('running job')
submit_utils.run_dicts(
    ks_final, param_combos_final, cmd_python=cmd_python,
    script_name='03_train_prefix.py', actually_run=True,
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
