import itertools
import os
from os.path import dirname

repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python3 01_train.py --prefix_or_suffix suffix --save_dir /home/chansingh/mntv1/sweep1 --checkpoint gpt2-medium --batch_size 200

# slurm params
partition = 'amlk8s'
num_gpus = 1
time = '4-0'
# s = Slurm("embed_dset", {"partition": partition, "time": time, "gres": f"gpu:{num_gpus}"})


##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    'seed': [1],
    'n_shots': [1, 3],
    'max_digit': [10],
    'beam_width_suffix': [5],
    'prefix_or_suffix': ['suffix'],
    'save_dir': ['/home/chansingh/mntv1/sweep1'],
    'task': ['add_two', 'first_two'],
}


##########################################
# params that are coupled together
##########################################
PARAMS_COUPLED_DICT = { # these batch_sizes are roughly set for an A100 80GB gpu
    ('checkpoint', 'batch_size'): [('gpt2-medium', 200), ('gpt2-large', 100),
                                   ('gpt2-xl', 40), ('EleutherAI/gpt-j-6B', 40)],
}

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
param_combos_final = [shared + combo
                      for shared in param_tuples_list_shared
                      for combo in param_tuples_list_coupled_flattened]
ks_final = ks_shared + list(sum(ks_coupled, ()))


for i in range(len(param_combos_final)):
    param_str = '/usr/bin/python3 ' + os.path.join(repo_dir, '01_train.py ')
    for j, key in enumerate(ks_final):
        param_str += '--' + key + ' ' + str(param_combos_final[i][j]) + ' '
    print(param_str)
    # s.run(param_str)
    # os.system(param_str)
