import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python3 02_train.py --prefix_or_suffix suffix --save_dir /home/chansingh/mntv1/sweep1 --checkpoint gpt2-medium --batch_size 200
# python3 02_train.py --save_dir /home/chansingh/mntv1/test
#  /usr/bin/python3 /home/chansingh/auto-prompt-engineering/01_train.py --use_single_query 1 --n_shots 1 --task add_two --use_parallelformers 0 --use_cpu_only 1 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --prefix_or_suffix suffix --save_dir /home/chansingh/mntv1/single_query6 --checkpoint gpt2-medium --batch_size 200

if len(sys.argv) > 1:
    print('running in amlt mode...')
    cmd_python = 'python'
    save_dir = '/mnt/output/single_query_math_9_12'  # sys.argv[1]
    assert save_dir.startswith('/mnt/output'), 'need to save to mount'
else:
    save_dir = '/home/chansingh/mntv1/single_query_math_9_12'
    cmd_python = 'python'

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things we vary
    'use_single_query': [1],
    'n_shots': [1, 5, 10],
    'task': ['add_two', 'multiply_two', 'divide_two', 'subtract_two',
             'max_two', 'first_two',
             'square_one', 'exp_one', 'double_one', 'fibonacci_one'],

    # parallel settings
    'use_parallelformers': [0],
    'use_cpu_only': [1],

    # things to average over
    'seed': [1, 2, 3],
    'template_num_init_string': [0, 1, 2],
    'template_num_task_phrasing': [0, 1, 2],

    # fixed params
    'max_digit': [10],
    'beam_size': [5],
    'save_dir': [save_dir],
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
        # ('EleutherAI/gpt-neox-20b', 10),
    ],
}


ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

for i in range(len(param_combos_final)):
    param_str = cmd_python + ' ' + \
        os.path.join(repo_dir, '02_train_suffix.py ')
    for j, key in enumerate(ks_final):
        param_str += '--' + key + ' ' + str(param_combos_final[i][j]) + ' '
    print(
        f'\n\n-------------------{i + 1}/{len(param_combos_final)}--------------------\n', param_str)
    try:
        os.system(param_str)
        pass
    except Exception as e:
        print(e)
