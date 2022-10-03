import pickle as pkl
from iprompt.prompt_classification import create_model, test_model_on_task_with_prefix
from iprompt.data import get_data, TASKS
from tqdm import tqdm
import torch
import numpy as np
import argparse
from os.path import join as oj
import sys
import os
sys.path.append('..')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


if __name__ == '__main__':
    args_dict = {
        'max_dset_size': 10_000,
        'template_num_task_phrasing': 0,
        'n_shots': 1,
        'max_digit': 10,
    }
    args = argparse.Namespace(**args_dict)

    # model_name = 'EleutherAI/gpt-j-6B'  # 'EleutherAI/gpt-neox-20b'
    model_name = 'EleutherAI/gpt-neo-2.7B'
    # model_name = 'gpt2-xl'  # 'EleutherAI/gpt-neox-20b'
    # model_name = 'gpt2-medium'  # 'EleutherAI/gpt-neox-20b'
    # model_name = 'gpt3'
    # model_name = 'facebook/opt-1.3b'
    # model_name = 'EleutherAI/gpt-neox-20b'
    save_dir = 'results/prompt_classification'
    # task_names = TASKS.keys()
    task_names = ['add_two', 'divide_two', 'max_two',
                  'subtract_two', 'first_two', 'multiply_two']
    # task_names = list(set(TASKS.keys()) - {'SUFFIXES'})
    batch_size = 1
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up tasks
    if model_name == 'gpt3':
        print('*** Warning: GPT-3 specified, so making calls to GPT-3 API ***')
    print('set up tasks')
    task_information = {}
    task_descriptions = []
    for name in task_names:
        dset, check_answer_func, description = get_data(
            args=args, task_name=name)
        task_descriptions.append(description)

    # load stuff
    print(f'load model {model_name}...')
    model = create_model(model_name)

    # calc losses & accs
    print('running...')
    losses = np.zeros((len(task_names), len(task_descriptions)))
    accuracies = np.zeros((len(task_names), len(task_descriptions)))
    for i in tqdm(range(len(task_names))):
        name = task_names[i]
        for j in range(len(task_descriptions)):
            description = task_descriptions[j]
            dset, check_answer_func, __this_task_description = get_data(
                args=args, task_name=name)
            loss, acc = test_model_on_task_with_prefix(
                dset=dset, model=model, prefix=f'{description} '
            )

            # save things
            losses[i][j] += loss
            accuracies[i][j] += acc
            # print(f'Task: {name} \t Prefix: {description} \t Loss: {loss:.2f}')

    # save
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump({
        'task_names': task_names,
        'task_descriptions': task_descriptions,
        'losses': losses,
        'accuracies': accuracies},
        open(oj(save_dir, f'heatmap_{model_name.replace("/", "___")}.pkl'), 'wb'))
