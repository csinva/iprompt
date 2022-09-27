import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

save_dir = '/home/johnmorris/interpretable-autoprompting/results/autoprompt_sentiment/tweets'

cmd_python = 'python'


# python 03_train_prefix.py --model_cls genetic --num_learned_tokens 16 --seed 3 --max_n_datapoints 40000 
# --early_stopping_steps 100 --checkpoint EleutherAI/gpt-j-6B --batch_size 32 --float16 1 --n_shots=5 
# --single_shot_loss=1   --mask_possible_answers=0   --task_name sst2_train
PARAMS_SHARED_DICT = {
    # things to vary
    'mask_possible_answers': [0],
    'model_cls': [
        'autoprompt',
        'genetic'
    ],
    'num_learned_tokens': [16],
    'task_name_list': [
        # 'sst2_train',
        # 'imdb_train',
        # 'rt_train',
        # 'ffb_train',
        'tweets_train',
    ],

    # iprompt_generation_repetition_penalty: [1.0, 1.5, 2.0],

    # stopping criteria
    'max_dset_size': [10_000], # sst2 has 10k sentences but could be more with a higher n_shots.
    'max_n_datapoints': [10_000],
    'early_stopping_steps': [50],

    # fixed params
    'train_split_frac': [0.75], # since some don't have a test set
    'single_shot_loss': [1],
    'n_shots': [5],
    'seed': [
        3,
        2,
        1,
    ],
    'max_length': [128],
    'iprompt_generation_repetition_penalty': [1.0],
}
PARAMS_SHARED_DICT['save_dir'] = [save_dir]

PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size', 'float16'): [
        ('EleutherAI/gpt-j-6B', 8, 1)
    ],
    # things to average over
    # ('seed', 'iprompt_generation_repetition_penalty'): [
    #     (1, 1.1),
    #     (2, 1.5),
    #     (3, 2.0),
    # ]
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

print('running job')
submit_utils.run_dicts(
    ks_final, param_combos_final, cmd_python=cmd_python,
    script_name='03_train_prefix.py', actually_run=True,
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
