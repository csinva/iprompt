import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python3 01_train.py --prefix_or_suffix suffix --save_dir /home/chansingh/mntv1/sweep1 --checkpoint gpt2-medium --batch_size 200
# python3 01_train.py --save_dir /home/chansingh/mntv1/test

if len(sys.argv) > 1:
    print('running in amlt mode...')
    cmd_python = 'python'
    save_dir = sys.argv[1]
    assert save_dir.startswith('/mnt/output'), 'need to save to mount'
else:
    save_dir = '/home/chansingh/mntv1/single_query'
    cmd_python = '/usr/bin/python3'

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    'single_query': [1],
    'n_shots': [1, 5, 10],
    'seed': [1],
    'max_digit': [10],
    'beam_width_suffix': [5],
    'prefix_or_suffix': ['suffix'],
    'save_dir': [save_dir],
    'task': ['add_two', 'multiply_two', 'divide_two', 'subtract_two', 'max_two'],
    'template_num_init_prefix': [0, 1],
    'template_num_task_phrasing': [0, 1],
}


##########################################
# params that are coupled together
##########################################
PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size'): [
        ('gpt2-medium', 200),
        ('gpt2-large', 100),
        ('gpt2-xl', 40),
        # ('EleutherAI/gpt-j-6B', 40)
        ('EleutherAI/gpt-neox-20b', 10),
    ],
}


ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

for i in range(len(param_combos_final)):
    param_str = cmd_python + ' ' + os.path.join(repo_dir, '01_train.py ')
    for j, key in enumerate(ks_final):
        param_str += '--' + key + ' ' + str(param_combos_final[i][j]) + ' '
    print(
        f'\n\n-------------------{i + 1}/{len(param_combos_final)}--------------------\n', param_str)
    try:
        os.system(param_str)
        pass
    except Exception as e:
        print(e)
