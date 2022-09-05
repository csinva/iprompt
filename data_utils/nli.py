import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

NLI_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'nli_processed')
DESCRIPTIONS_DICT = json.load(open(
    oj(NLI_PROCESSED_DIR, 'task_defs.json'), 'r'))


def fetch_data(task_name_nli):
    return pd.read_csv(oj(NLI_PROCESSED_DIR, task_name_nli + '.csv'))


TASKS_NLI = {
    'task1146_country_capital': {
        'check_answer_func': r'capital',
    },
    'task1147_country_currency': {
        'check_answer_func': r'currency|money',
    },
    'task1149_item_check_edible': {
        'check_answer_func': r'edible|eatable|safe to eat',
    },
    'SUFFIXES': ['What is the task defined above?'],
}
ks = list(TASKS_NLI.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_NLI[k]['description'] = DESCRIPTIONS_DICT[k]
        TASKS_NLI[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_NLI)
