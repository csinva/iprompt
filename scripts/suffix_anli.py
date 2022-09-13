import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python ../02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0 --use_single_query 0 --use_stopwords 0
# python ../02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0 --use_single_query 1 --use_stopwords 0
# python /home/chansingh/interpretable-autoprompting/02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0 --use_single_query 1
# python /home/chansingh/interpretable-autoprompting/02_train_suffix.py --n_shots 5 --task task1146_country_capital --use_parallelformers 0 --use_cpu_only 0 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200
# python /home/chansingh/interpretable-autoprompting/02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0

if len(sys.argv) > 1:
    print('running in amlt mode...')
    cmd_python = 'python'
    save_dir = '/mnt/output/suffix_anli_9_12'  # sys.argv[1]
    assert save_dir.startswith('/mnt/output'), 'need to save to mount'
else:
    save_dir = '/home/chansingh/mntv1/suffix_anli_9_12'
    cmd_python = '/home/chansingh/.autoprompt/bin/python' 

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to vary
    'n_shots': [1, 5],
    'task': ['task1146_country_capital', 'task1509_evalution_antonyms', 'task1147_country_currency',
             'task1149_item_check_edible', 'task183_rhyme_generation', 'task1191_food_veg_nonveg',
             'task092_check_prime_classification', 'task088_identify_typo_verification',
             'task1336_peixian_equity_evaluation_corpus_gender_classifier', 'task107_splash_question_to_sql'],

    # parallel settings
    'use_parallelformers': [0],
    'use_cpu_only': [0],

    # things to average over
    'seed': [1],
    'template_num_init_string': [0],
    'template_num_task_phrasing': [0],

    # fixed params
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
