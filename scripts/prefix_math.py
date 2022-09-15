import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/interpretable-autoprompting/01_train_suffix.py --n_shots 5 --task task1146_country_capital --use_parallelformers 0 --use_cpu_only 0 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-xl --batch_size 200
# python /home/chansingh/interpretable-autoprompting/01_train_suffix.py --n_shots 3 --task task1146_country_capital --use_parallelformers 0 --use_cpu_only 0 --seed 1 --template_num_init_string 0 --template_num_task_phrasing 0 --max_digit 10 --beam_width_suffix 5 --save_dir /home/chansingh/mntv1/sweep_anli_9_12 --checkpoint gpt2-medium --batch_size 200

save_dir = f'/home/chansingh/mntv1/prefix_math_{submit_utils.JOB_SUFFIX}'
cmd_python = 'python'

from suffix_math import PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT

PARAMS_SHARED_DICT.update({
    'mlm_num_candidates': 256,
    'do_reranking': [0, 1],
})

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

submit_utils.run_dicts(ks_final, param_combos_final, cmd_python=cmd_python,
                       script_name='03_find_prefix_rerank.py', actually_run=False
)
