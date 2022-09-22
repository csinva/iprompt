import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# save_dir = f'/home/chansingh/mntv1/prefix_math_{submit_utils.JOB_SUFFIX}'
# save_dir = f'/home/jxm3/random/interpretable-autoprompting/results/tst2/prefix_math_{submit_utils.JOB_SUFFIX}'
save_dir = '/home/jxm3/random/interpretable-autoprompting/results/autoprompt_add_test'

cmd_python = 'python'

PARAMS_SHARED_DICT = {
    # things to vary
    'n_shots': [1],
    task_name_list = [['add_two']],
    # 'task_name_list': [['add_two', 'multiply_two', 'divide_two', 'subtract_two',
    #          'max_two', 'first_two',
    #          'square_one', 'exp_one', 'double_one', 'fibonacci_one']],

    # things to average over
    'seed': [1],
    'template_num_init_string': [0], #, 1, 2],
    'template_num_task_phrasing': [0], #, 1, 2],

    # fixed params
    'max_digit': [10],
}
PARAMS_SHARED_DICT.update(submit_utils.PARAMS_SHARED_DICT_PREFIX)
PARAMS_SHARED_DICT['save_dir'] = [save_dir]

# # Temp stuff: only need to re-run 5-shot experiments,
# # with reranking, with max 64 examples.
# PARAMS_SHARED_DICT['do_reranking'] = [0, 1]
# PARAMS_SHARED_DICT['n_shots'] = [5]
# PARAMS_SHARED_DICT['max_num_samples'] = [64]

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, submit_utils.PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='03_rerank_prefix.py', actually_run=True
)
