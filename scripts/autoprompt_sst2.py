import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

save_dir = '/home/johnmorris/interpretable-autoprompting/results/autoprompt_sst2'

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1, 2],

    # things to vary
    'n_shots': [1],
    'task_name_list': ['sst2_train'],
    'model_cls': ['genetic', 'autoprompt'],
    'num_learned_tokens': [16],

    # stopping criteria
    'max_dset_size': [20_000], # sst2 has 10k sentences but could be more with a higher n_shots.
    'max_n_datapoints': [20_000],
    'early_stopping_steps': [50],

    # fixed params
    'train_split_frac': [1.0],
    'single_shot_loss': [1],
}
PARAMS_SHARED_DICT['save_dir'] = [save_dir]

PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size', 'float16'): [
        ('EleutherAI/gpt-j-6B', 32, 1)
    ],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

print('running job')
submit_utils.run_dicts(
    ks_final, param_combos_final, cmd_python=cmd_python,
    script_name='03_train_prefix.py', actually_run=True,
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
