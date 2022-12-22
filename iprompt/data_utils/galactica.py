import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

GALACTICA_PROCESSED_DIR = oj(
    dirname(os.path.abspath(__file__)), 'galactica_raw')


def fetch_bbbp():
    df = pd.read_csv(oj(GALACTICA_PROCESSED_DIR, 'bbbp.csv'))
    df.name.loc[df.name.str.isnumeric()] = 'Compound-'+df.name.loc[df.name.str.isnumeric()] # rename compounds that are just numbers
    # Fix input: Encourage model to answer output as next token.
    df['input'] = df['name'].map(lambda s: f'Here is a compound:\n{s}') + '\nAnswer:'
    # df['input'] = df['smiles'].map(lambda s: f'Here is a compound:\n[START_I_SMILES]{s}[END_I_SMILES]') + '\nAnswer:'
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    # df['output'] = df['p_np'].map({0: 'NP', 1: 'P'}).map(lambda s: f' {s}\n\n')
    df['output'] = df['p_np'].map({0: 'No', 1: 'Yes'}).map(lambda s: f' {s}.\n\n')
    return df

TASKS_GALACTICA = {
    'bbbp': {
        'check_answer_func': r'blood|brain|barrier|permeability|bbb',
        'description': 'Check if the compound permeates the blood-brain barrier',
        'gen_func': fetch_bbbp,
    }
}
ks = list(TASKS_GALACTICA.keys())
        

if __name__ == '__main__':
    print(TASKS_GALACTICA)
    task = TASKS_GALACTICA['bbbp']
    df = task['gen_func']()
    print(repr(df.iloc[0]))
    print(repr(df.iloc[0]['input']))
    print(repr(df.iloc[0]['output']))
    print(df.head())
