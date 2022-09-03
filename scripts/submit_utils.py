import itertools
import os
from os.path import dirname
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))

def combine_param_dicts(PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT):

    # shared
    ks_shared = list(PARAMS_SHARED_DICT.keys())
    vals_shared = [PARAMS_SHARED_DICT[k] for k in ks_shared]
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