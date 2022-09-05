import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

NLI_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'nli_processed')
DESCRIPTIONS_DICT = json.load(open(
    oj(NLI_PROCESSED_DIR, 'task_defs_brief.json'), 'r'))
    # oj(NLI_PROCESSED_DIR, 'task_defs.json'), 'r'))

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
    'task1149_item_check_edible': {
        'check_answer_func': r'edible|eatable|safe to eat',
    },
    'task429_senteval_tense': {
        'check_answer_func': r'tense|past|present',
    },
    'task430_senteval_subject_count': {
        'check_answer_func': r'singular|plural', # this is shaky
    },
    'task609_sbic_potentially_offense_binary_classification': {
        'check_answer_func': r'offensive|toxic|harmful|derogatory|hate speech', # this is shaky
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
