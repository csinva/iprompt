import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/interpretable-autoprompting/01_train_suffix.py --n_shots 5 --task task1146_country_capital --use_parallelformers 0 --use_cpu_only 0 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200
# python /home/chansingh/interpretable-autoprompting/01_train_suffix.py --n_shots 3 --task task1146_country_capital --use_parallelformers 0 --use_cpu_only 0 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-medium --batch_size 200

"""
if len(sys.argv) > 1:
    print('running in amlt mode...')
    # save_dir = '/mnt/output/suffix_math_9_15'  # sys.argv[1]
    # assert save_dir.startswith('/mnt/output'), 'need to save to mount'
else:
    # save_dir = '/home/chansingh/mntv1/suffix_math_9_16'
    save_dir = f'/home/chansingh/mntv1/suffix_math_{submit_utils.JOB_SUFFIX}'
"""

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = submit_utils.PARAMS_SHARED_DICT_MATH
PARAMS_SHARED_DICT.update(submit_utils.PARAMS_SHARED_DICT_SUFFIX)
##########################################
# params that are coupled together
##########################################
PARAMS_COUPLED_DICT = submit_utils.PARAMS_COUPLED_DICT

# default
# PARAMS_SHARED_DICT['save_dir'] = ['/home/chansingh/mntv1/suffix_math_9_17']

# long_suffs
# PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_math_long_suffs']
# PARAMS_SHARED_DICT['max_num_tokens'] = [4] 

# double_digit
# PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_math_double_digit']
# PARAMS_SHARED_DICT['max_digit'] = [100] 

# long_suffs_full
PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_math_long_suffs_full']
PARAMS_SHARED_DICT['max_num_tokens'] = [3, 6]
PARAMS_SHARED_DICT['use_early_stopping'] = [0]
PARAMS_SHARED_DICT['use_generic_query'] = [1]
PARAMS_SHARED_DICT['n_shots'] = [1, 5]

cmd_python = 'python'
ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)
submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='02_train_suffix.py', actually_run=True)