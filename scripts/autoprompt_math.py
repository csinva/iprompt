import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/prefix_math_{submit_utils.JOB_SUFFIX}'
# save_dir = f'/home/jxm3/random/interpretable-autoprompting/results/tst2/prefix_math_{submit_utils.JOB_SUFFIX}'
# save_dir = '/home/jxm3/random/interpretable-autoprompting/results/slurm_math_exps'
save_dir = '/home/johnmorris/interpretable-autoprompting/results/autoprompt_arithmetic'

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1],

    # things to vary
    'n_shots': [1, 5],

    # 'task_name_list': [['add_two']],
    'task_name_list': [
        # 'add_two', 'multiply_two', 
        # 'subtract_two',
        # 'max_two', 'first_two',
        # 'square_one', 'double_one',
        'exp_one',  'fibonacci_one',
        'divide_two', 
    ],
    'model_cls': ['genetic', 'autoprompt'],
    'num_learned_tokens': [3, 6],

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

PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size', 'float16'): [
        # ('gpt2', 32, 0),
        # ('gpt2-medium', 200, 0),
        # ('gpt2-large', 100, 0),
        # ('gpt2-xl', 32, 0),
        # ('EleutherAI/gpt-neo-2.7B', 16, 0),
        ('EleutherAI/gpt-j-6B', 32, 1)
        # ('EleutherAI/gpt-neox-20b', 1, 0),
    ],
}

## ***** tmp, for testing
# PARAMS_SHARED_DICT['n_shots'] = [1]
# PARAMS_SHARED_DICT['num_learned_tokens'] = [3]
# PARAMS_SHARED_DICT['task_name_list'] = ['exp_one']
# PARAMS_SHARED_DICT['max_n_datapoints'] = ['50']
## ******

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

print('running job')
submit_utils.run_dicts(
    ks_final, param_combos_final, cmd_python=cmd_python,
    script_name='03_train_prefix.py', actually_run=True,
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
