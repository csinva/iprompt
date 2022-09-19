import itertools
import os
from os.path import dirname
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))

JOB_SUFFIX = 'long_suffs'
PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size'): [
        # ('gpt2', 32),
        # ('gpt2-medium', 200),
        # ('gpt2-large', 100),
        ('gpt2-xl', 40),
        ('EleutherAI/gpt-neo-2.7B', 20),
        ('EleutherAI/gpt-j-6B', 10)
        # ('EleutherAI/gpt-neox-20b', 1),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT_MATH = {
    # things to vary
    'n_shots': [1, 5, 10],
    'task_name_list': [['add_two', 'multiply_two', 'divide_two', 'subtract_two',
             'max_two', 'first_two',
             'square_one', 'exp_one', 'double_one', 'fibonacci_one']],

    # things to average over
    'seed': [1],
    'template_num_init_string': [0], #, 1, 2],
    'template_num_task_phrasing': [0], #, 1, 2],

    # fixed params
    'max_digit': [10],
}

PARAMS_SHARED_DICT_ANLI = {
    # things to vary
    'n_shots': [1, 5],
    'task_name_list': [['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
             'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
             'task092_check_prime_classification', 'task088_identify_typo_verification',
             'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql']],

    # things to average over
    'seed': [1],
    'template_num_init_string': [0],
    'template_num_task_phrasing': [0],
}

PARAMS_SHARED_DICT_SUFFIX = {
    # fixed params
    'beam_size': [5],
    'beam_size_extra': [50],
    'max_num_tokens': [1],
    # parallel settings
    'use_parallelformers': [0],
    'use_cpu_only': [0],
}

PARAMS_SHARED_DICT_PREFIX = {
    # 'mlm_num_candidates': [256],
    'mlm_num_candidates': [128],
    'do_reranking': [0, 1],
    'max_num_samples': [0, 1], # try full dataset (0) and single-sample (1)
}

def combine_param_dicts(PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT):
    # shared
    ks_shared = list(PARAMS_SHARED_DICT.keys())
    vals_shared = [PARAMS_SHARED_DICT[k] for k in ks_shared]
    for val in vals_shared:
        assert isinstance(val, list), f"param val {val} must be type list, got type {type(val)}"
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

