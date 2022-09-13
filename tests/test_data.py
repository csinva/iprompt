import os
from os.path import dirname
import sys
import pandas as pd
from os.path import dirname
from os.path import join as oj
REPO_DIR = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)
from data_utils.anli import TASKS_ANLI
from data import get_data
from data_utils import data_funcs
from data_utils.one_num import TASKS_ONE_NUM
from data_utils.two_num import TASKS_TWO_NUMS
from data_utils.anli import TASKS_ANLI
from data import TASKS

def test_data():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value = os.system('python ' + os.path.join(repo_dir, 'data.py '))
    assert exit_value == 0, 'default data loading passed'

def test_get_data():
    print('\n################Lots of available dsets############\n')
    for task_key in TASKS:
        if not task_key == 'SUFFIXES':
            # , TASKS[task_key]['description'])
            print(task_key, '->', TASKS[task_key]['description'] + '\n')

    print('\n\n################Lets look at some examples############\n')

    class fake_args:
        template_num_task_phrasing = 0
        max_dset_size = 1000
        max_digit = 10
    task_name = 'multiply_two'

    args = fake_args()
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=3)
    print('Example 3-shot (max_digit=10)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    args.max_digit = 100
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print('Example 1-shot (max_digit=100)', repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name='fibonacci_one', n_shots=1)
    print('Example fibonacci_one 1-shot (max_digit=10)',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    print('\n\n################Lets look at an ANLI dataset############\n')
    task_name = 'task1147_country_currency'
    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=1)
    print(f'Example {task_name} 1-shot',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    dset, check_answer_func, descr = get_data(
        args, task_name=task_name, n_shots=3)
    print(f'Example {task_name} 3-shot',
          repr(dset[0]['text']))
    print('\tlen', len(dset))

    print('\n\n################Lets look at how answers are checked############\n')
    task_name = 'add_two'
    task = TASKS[task_name]
    _, check_answer_func, descr = get_data(args, task_name=task_name)
    print('checking func', check_answer_func, 'for', task_name)
    for s in ['add', 'take the nums and add', 'test', ' sum', 'Add']:
        print(repr(s), check_answer_func(s))

def test_anli():
    class fake_args:
        template_num_task_phrasing = 0
        max_dset_size = 1000
        max_digit = 10
    args = fake_args()
    for task_name in TASKS_ANLI:
        if not task_name == 'SUFFIXES':
            print(task_name)
            df, answer_func, descr = get_data(args, task_name)
            def check_text(s):
                return isinstance(s, str) and len(s) > 0
            assert pd.Series(df['text']).apply(
                check_text).all(), 'text is all strings'
            assert check_text(descr)

if __name__ == '__main__':
    test_get_data()