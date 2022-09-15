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
    df = pd.read_csv(oj(ANLI_PROCESSED_DIR, task_name_anli + '.csv'))
    # Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    return df


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
    'task1509_evalution_antonyms': {  # Given a word generate its antonym
        'check_answer_func': r'opposite|antonym',
    },
    'task183_rhyme_generation': { 
        'check_answer_func': r'rhyme|rhy',
    },
    'task1191_food_veg_nonveg': {
        'check_answer_func': r'vegetarian|vegan|meat',
    },
    'task092_check_prime_classification': {
        'check_answer_func': r'prime',
    },
    'task107_splash_question_to_sql': {
        'check_answer_func': r'sql',
    },
    'task1336_peixian_equity_evaluation_corpus_gender_classifier': {
        'check_answer_func': r'gender|fem|masc',
    },
    'task088_identify_typo_verification': {
        'check_answer_func': r'mistake|typo|mistype|spell',
    },
    # 'task1321_country_continent': { # Given a word generate its antonym
    # 'check_answer_func': r'edible|eatable|safe to eat',
    # },    
    # 'task429_senteval_tense': {
    # 'check_answer_func': r'tense|past|present',
    # },
    # 'task430_senteval_subject_count': {
    # 'check_answer_func': r'singular|plural', # this is shaky
    # },
    # 'task609_sbic_potentially_offense_binary_classification': {
    # 'check_answer_func': r'offensive|toxic|harmful|derogatory|hate speech', # this is shaky
    # },

    # generic suffixes don't work
    # SUFFIXES = [
    # 'How do we get the answer from the input?\nThe answer is'
    # 'The answer is the'
    # 'To get the answer from the input,',
    # 'How do we get the answer from the input?\nTo get the answer, we'
    # ]
    'SUFFIXES': {
        'task1146_country_capital': ['Given the input country, the answer is the country\'s'],
        'task1147_country_currency': ['Given the input country, the answer is the country\'s'],
        'task1149_item_check_edible': ['The answer is whether or not the input item is'],
        'task1509_evalution_antonyms': ['The answer takes the input word and returns its'],
        'task183_rhyme_generation': ['The answer takes the input word and returns its'],
        'task1191_food_veg_nonveg': ['The answer takes the food and returns whether is is'],
        'task092_check_prime_classification': ['The answer takes the input number and returns whether it is'],
        'task107_splash_question_to_sql': ['The answer takes the input word and returns it as a'],
        'task1336_peixian_equity_evaluation_corpus_gender_classifier': ['The answer takes the input and returns the subject\'s'],
        'task088_identify_typo_verification': ['The answer is the word in the input that is a'],
    },
}
ks = list(TASKS_ANLI.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_ANLI[k]['description'] = DESCRIPTIONS_DICT[k]
        TASKS_ANLI[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_ANLI)
