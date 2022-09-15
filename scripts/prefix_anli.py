import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

save_dir = f'/home/chansingh/mntv1/prefix_anli_{submit_utils.JOB_SUFFIX}'
cmd_python = 'python'

from suffix_math import PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT


PARAMS_SHARED_DICT = submit_utils.PARAMS_SHARED_DICT_ANLI
PARAMS_SHARED_DICT.update({
    'mlm_num_candidates': 256,
    'do_reranking': [0, 1],
    'single_query': [0, 1],
    'save_dir': [save_dir],
})

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, submit_utils.PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='03_find_prefix_rerank.py', actually_run=False
)
