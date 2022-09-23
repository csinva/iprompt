import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/prefix_math_{submit_utils.JOB_SUFFIX}'
# save_dir = f'/home/jxm3/random/interpretable-autoprompting/results/tst2/prefix_math_{submit_utils.JOB_SUFFIX}'
save_dir = '/home/jxm3/random/interpretable-autoprompting/results/autoprompt_add_test_2'

cmd_python = 'python'


"""
example_command:

torch ❯ python 03_train_prefix.py  --use_preprefix=0 --num_learned_tokens=8 
--batch_size=16  --accum_grad_over_epoch=0 --model_cls=genetic 
--checkpoint="EleutherAI/gpt-j-6B" --n_shots=1 --seed=1  
--task_name add_two --max_n_steps=500 --train_split_frac=0.8 
--max_digit=100 --float16=1 
--early_stopping_steps=10
"""
PARAMS_SHARED_DICT = {
    # things to vary
    'n_shots': [1, 5],
    # 'task_name_list': [['add_two']],
    'task_name_list': [
        'add_two', 'multiply_two', 
        'subtract_two',
        'max_two', 'first_two',
        'square_one', 'double_one',
    ], # 'exp_one',  'fibonacci_one', 'divide_two', 
    'model_cls': ['genetic', 'autoprompt'],
    'num_learned_tokens': [3, 6],

    # things to average over
    'seed': [1],
    # TODO: average over pre-prompt/post-prompt for genetic,
    # possibly initialization  or num candidates for autoprompt.

    # stopping criteria
    'max_dset_size': [1000],
    'max_n_datapoints': [4000],
    'early_stopping_steps': [20],

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


ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='03_train_prefix.py', actually_run=False
)
