import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python ../02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0 --use_single_query 0 --use_stopwords 0
# python ../02_train_suffix.py --n_shots 3 --task task1146_country_capital --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200 --use_cache 0 --use_single_query 1 --use_stopwords 0

"""
if len(sys.argv) > 1:
    print('running in amlt mode...')
    # save_dir = '/mnt/output/suffix_anli_9_14'  # sys.argv[1]
    # assert save_dir.startswith('/mnt/output'), 'need to save to mount'
else:
    save_dir = f'/home/chansingh/mntv1/suffix_anli_{submit_utils.JOB_SUFFIX}'
"""

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = submit_utils.PARAMS_SHARED_DICT_ANLI
PARAMS_SHARED_DICT.update(submit_utils.PARAMS_SHARED_DICT_SUFFIX)

##########################################
# params that are coupled together
##########################################
PARAMS_COUPLED_DICT = submit_utils.PARAMS_COUPLED_DICT

# default
# PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_anli_9_17']

# long_suffs
# PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_anli_long_suffs']
# PARAMS_SHARED_DICT['max_num_tokens'] = [4] 

# long_suffs_full
PARAMS_SHARED_DICT['save_dir'] = [f'/home/chansingh/mntv1/suffix_long_rerun']
PARAMS_SHARED_DICT['max_num_tokens'] = [3, 6]
PARAMS_SHARED_DICT['use_early_stopping'] = [0]
PARAMS_SHARED_DICT['use_generic_query'] = [1]
PARAMS_SHARED_DICT['n_shots'] = [1]
PARAMS_SHARED_DICT['train_split_frac'] = [0.75]
PARAMS_SHARED_DICT['float16'] = [1]
PARAMS_SHARED_DICT['max_dset_size'] = [5000]
PARAMS_COUPLED_DICT = {  # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size'): [
        ('EleutherAI/gpt-j-6B', 8)
    ],
}


cmd_python = 'python'
ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='02_train_suffix.py', actually_run=True,
                       use_slurm=False, save_dir=PARAMS_SHARED_DICT['save_dir']
)
