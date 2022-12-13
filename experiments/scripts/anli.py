import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

save_dir = f'/home/chansingh/mntv1/iprompt_revision2/anli/'

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1],

    # things to vary
    'n_shots': [1, 5],
    'task_name_list': [
        'task1146_country_capital',
        'task1147_country_currency',
        'task1509_evalution_antonyms',
        'task1149_item_check_edible',
        'task183_rhyme_generation',
        'task1191_food_veg_nonveg',
        'task092_check_prime_classification',
        'task088_identify_typo_verification',
        'task1336_peixian_equity_evaluation_corpus_gender_classifier',
        'task107_splash_question_to_sql'
    ],
    'model_cls': ['autoprompt'],
    'num_learned_tokens': [6, 12],

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
    use_slurm=False, save_dir=save_dir, slurm_gpu_str='gpu:a6000:1',
)
