import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

INDUCTION_PROCESSED_DIR = oj(
    dirname(os.path.abspath(__file__)), 'induction_processed')
DESCRIPTIONS_DICT = json.load(open(
    # oj(ANLI_PROCESSED_DIR, 'task_defs_brief.json'), 'r'))
    oj(INDUCTION_PROCESSED_DIR, 'task_defs.json'), 'r')
)


def fetch_data(task_name_induction):
    df = pd.read_csv(oj(INDUCTION_PROCESSED_DIR, task_name_induction + '.csv'))
    # Fix input: Encourage model to answer output as next token.
    df['input'] = df['input'].map(lambda s: f'Input: {s} Answer:')
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    return df


TASKS_INDUCTION = {
    'cause_and_effect': {
        'check_answer_func': r'caus|effect|reason',
    },
    'sum': {
        'check_answer_func': r'add|sum',
    },
    'num_to_verbal': {
        'check_answer_func': r'number|numeric',
    },
    'diff': {
        'check_answer_func': r'difference|subtract',
    },
    'first_word_letter': {
        'check_answer_func': r'first',
    },
    'singular_to_plural': {
        'check_answer_func': r'sing|plural',
    },
    'synonyms': {
        'check_answer_func': r'synonym|alternate|rephrase',
    },
    'letters_list': {
        'check_answer_func': r'list|sequence',
    },
    'sentence_similarity': {
        'check_answer_func': r'similarity|same|same meaning|same meaning as',
    },
    'informal_to_formal': {
        'check_answer_func': r'formal|informal|polite|impolite',
    },
    'rhymes': {
        'check_answer_func': r'rhyme',
    },
    'common_concept': {
        'check_answer_func': r'common|shared|same|concept|meaning|idea',
    },
    'second_word_letter': {
        'check_answer_func': r'second',
    },
    'translation_en-fr': {
        'check_answer_func': r'translat',
    },
    'taxonomy_animal': {
        'check_answer_func': r'animal',
    },
    'sentiment': {
        'check_answer_func': r'sentiment|positive|negative',
    },
    'active_to_passive': {
        'check_answer_func': r'active|passive',
    },
    'word_in_context': {
        'check_answer_func': r'same|same meaning|same meaning as',
    },
    'orthography_starts_with': {
        'check_answer_func': r'orthography|starts with',
    },
    'antonyms': {
        'check_answer_func': r'antonym|opposite',
    },
    'negation': {
        'check_answer_func': r'negation|opposite',
    },
    'translation_en-de': {
        'check_answer_func': r'translat',
    },
    'larger_animal': {
        'check_answer_func': r'large|larger|bigger|biggest',
    },
    'translation_en-es': {
        'check_answer_func': r'translat',
    },
}
ks = list(TASKS_INDUCTION.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_INDUCTION[k]['description'] = DESCRIPTIONS_DICT[k]
        TASKS_INDUCTION[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_INDUCTION)
