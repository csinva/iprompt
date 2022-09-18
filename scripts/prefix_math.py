import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/prefix_math_{submit_utils.JOB_SUFFIX}'
save_dir = f'/home/jxm3/random/interpretable-autoprompting/results/tst2/prefix_math_{submit_utils.JOB_SUFFIX}'
cmd_python = 'python'

PARAMS_SHARED_DICT = submit_utils.PARAMS_SHARED_DICT_MATH
PARAMS_SHARED_DICT.update(submit_utils.PARAMS_SHARED_DICT_PREFIX)
PARAMS_SHARED_DICT['save_dir'] = [save_dir]

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, submit_utils.PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='03_rerank_prefix.py', actually_run=True
)
