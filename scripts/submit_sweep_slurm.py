import itertools

# slurm params
partition = 'amlk8s'
num_gpus = 1
time = '4-0'
# s = Slurm("embed_dset", {"partition": partition, "time": time, "gres": f"gpu:{num_gpus}"})


##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED = {
    'seed': [1],
    'n_shots': [1, 3],
    'max_digit': [10],
    'prefix_or_suffix': ['suffix'],
    'save_dir': ['/home/chansingh/mntv1/sweep0'],
    'task': ['add_two', 'first_two'],
}


##########################################
# params that are coupled together
##########################################
PARAMS_LIST_COUPLED = {
    ('checkpoint', 'batch_size'): [('gpt2-medium', 100), ('gpt2-large', 50),
                                   ('gpt2-xl', 20), ('EleutherAI/gpt-j-6B', 20), ],
    
}


for PARAMS in PARAMS_LIST_COUPLED:
    ks = list(PARAMS.keys())
    vals = [PARAMS[k] for k in ks]

    ks2 = list(PARAMS_SHARED.keys())
    vals += [PARAMS_SHARED[k] for k in ks2]
    ks += ks2

    param_combinations = list(itertools.product(*vals))  # list of tuples

    for i in range(len(param_combinations)):
        param_str = '/usr/bin/python3 ../01_train.py '
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        print(param_str)
        # s.run(param_str)
