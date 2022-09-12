import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

ANLI_PROCESSED_DIR = oj(dirname(os.path.abspath(__file__)), 'anli_processed')
DESCRIPTIONS_DICT = json.load(open(
    # oj(ANLI_PROCESSED_DIR, 'task_defs_brief.json'), 'r'))
    oj(ANLI_PROCESSED_DIR, 'task_defs.json'), 'r'))


def fetch_data(task_name_anli):
    return pd.read_csv(oj(ANLI_PROCESSED_DIR, task_name_anli + '.csv'))


TASKS_ANLI = {
    'task1146_country_capital': {
        'check_answer_func': r'capital',
    },
    'task1147_country_currency': {
        'check_answer_func': r'currency|money',
    },
    'task1149_item_check_edible': {
        'check_answer_func': r'edible|eatable|safe to eat',
    },
    # 'task1321_country_continent': { # Given a word generate its antonym
    # 'check_answer_func': r'edible|eatable|safe to eat',
    # },
    'task1509_evalution_antonyms': {  # Given a word generate its antonym
        'check_answer_func': r'opposite|antonym',
    },
    'task183_rhyme_generation': {  # Given a word generate its antonym
        'check_answer_func': r'rhyme',
    },
    'task1191_food_veg_nonveg': {  # Given a word generate its antonym
        'check_answer_func': r'vegetarian|vegan|meat',
    },
    'task092_check_prime_classification': {  # Given a word generate its antonym
        'check_answer_func': r'prime',
    },
    'task107_splash_question_to_sql': {  # Given a word generate its antonym
        'check_answer_func': r'sql',
    },
    'task1336_peixian_equity_evaluation_corpus_gender_classifier': {  # Given a word generate its antonym
        'check_answer_func': r'gender|fem|masc',
    },
    'task088_identify_typo_verification': {  # Given a word generate its antonym
        'check_answer_func': r'mistake|typo|mistype|spell',
    },
    # 'task429_senteval_tense': {
    # 'check_answer_func': r'tense|past|present',
    # },
    # 'task430_senteval_subject_count': {
    # 'check_answer_func': r'singular|plural', # this is shaky
    # },
    # 'task609_sbic_potentially_offense_binary_classification': {
    # 'check_answer_func': r'offensive|toxic|harmful|derogatory|hate speech', # this is shaky
    # },

    'SUFFIXES': [
        'How do we get the answer from the input?\nThe answer is'
        # 'The answer is the'
        # 'To get the answer from the input,',
        # 'How do we get the answer from the input?\nTo get the answer, we'
    ],
}
ks = list(TASKS_ANLI.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_ANLI[k]['description'] = DESCRIPTIONS_DICT[k]
        TASKS_ANLI[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_ANLI)
