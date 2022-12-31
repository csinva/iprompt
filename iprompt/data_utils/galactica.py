import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj
import warnings

GALACTICA_PROCESSED_DIR = oj(
    dirname(os.path.abspath(__file__)), 'galactica_raw')


def get_bbbp():
    df = pd.read_csv(oj(GALACTICA_PROCESSED_DIR, 'bbbp.csv'))
    df.name.loc[df.name.str.isnumeric()] = 'Compound-'+df.name.loc[df.name.str.isnumeric()] # rename compounds that are just numbers
    # Fix input: Encourage model to answer output as next token.
    df['input'] = df['name'].map(lambda s: f'Here is a compound:\n{s}') + '\nAnswer:'
    # df['input'] = df['smiles'].map(lambda s: f'Here is a compound:\n[START_I_SMILES]{s}[END_I_SMILES]') + '\nAnswer:'
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['p_np'].map({0: 'NP', 1: 'P'}).map(lambda s: f' {s}\n\n')
    # df['output'] = df['p_np'].map({0: 'No', 1: 'Yes'}).map(lambda s: f' {s}.\n\n')
    return df

def get_tox_dset(tox_target=0, return_tox_target_name=False):
    """Get the tox21 dataset for a given toxicity target (speficied as an int).

    Params
    ------
    tox_target: int
        The toxicity target must be in [0, 12).
    """
    d = pd.read_csv(oj(GALACTICA_PROCESSED_DIR, 'tox21.csv'))
    # smiles = df['smiles']
    # d = df.drop(columns=['smiles']).set_index('mol_id')
    control = d.loc[(d.sum(axis=1) == 0) & (d.isna().sum(axis=1) == 0)]
    tox = d.loc[(d.sum(axis=1) == 1) & (d.iloc[:, tox_target] == 1)]
    n = tox.shape[0]
    control = control.sample(n=n, random_state=42)
    df = pd.concat([tox, control])
    df['tox'] = 1
    df['tox'].iloc[n:] = 0
    df = df.sample(frac=1, random_state=42)
    df['input'] = df['mol_id'].map(lambda s: f'Here is a compound:\n{s}') + '\nAnswer:' 
    df['output'] = df['tox'].map({0: 'No', 1: 'Yes'}).map(lambda s: f' {s}.\n\n')
    if return_tox_target_name:
        return df, d.columns[tox_target]
    else:
        return df

def load_uniprot(keyword1='Cytoplasm', keyword2='Membrane'):
    df = pd.read_csv(oj(GALACTICA_PROCESSED_DIR, f'{keyword1}_{keyword2}.tsv'), sep='\t')
    # df.name.loc[df.name.str.isnumeric()] = 'Compound-'+df.name.loc[df.name.str.isnumeric()] # rename compounds that are just numbers
    # Fix input: Encourage model to answer output as next token.
    df['input'] = df['Sequence'].map(lambda s: f'Here is a protein sequence:\n{s}') + '\nAnswer:'
    # df['input'] = df['smiles'].map(lambda s: f'Here is a compound:\n[START_I_SMILES]{s}[END_I_SMILES]') + '\nAnswer:'
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['key1'].map({0: 'Yes', 1: 'No'}).map(lambda s: f' {s}\n\n')
    # df['output'] = df['p_np'].map({0: 'No', 1: 'Yes'}).map(lambda s: f' {s}.\n\n')
    return df

TASKS_GALACTICA = {
    'bbbp': {
        'check_answer_func': r'blood|brain|barrier|permeability|bbb',
        'description': 'Check if the compound permeates the blood-brain barrier',
        'gen_func': get_bbbp,
    },
    'uniprot_cytoplasm_membrane': {
        'check_answer_func': r'cytoplasm|membrane',
        'description': 'Check if the protein has the keywor cytoplasm or membrane',
        'gen_func': lambda: load_uniprot(keyword1='Cytoplasm', keyword2='Membrane'),
    },
    'uniprot_rna-binding_atp-binding': {
        'check_answer_func': r'rna|atp',
        'description': 'Check if the protein has the keywor rna-binding or atp-binding',
        'gen_func': lambda: load_uniprot(keyword1='RNA-binding', keyword2='ATP-binding'),
    }
}
for i in range(12):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, tox_name = get_tox_dset(i, return_tox_target_name=True)
    TASKS_GALACTICA[f'tox21_{i}'] = {
        'check_answer_func': rf'{tox_name}',
        'description': f'Distinguish whether a compound is toxic to {tox_name}',
        'gen_func': lambda: get_tox_dset(i, return_tox_target_name=False),
    }
ks = list(TASKS_GALACTICA.keys())
        

if __name__ == '__main__':
    print(TASKS_GALACTICA)
    # task = TASKS_GALACTICA['bbbp']
    task = TASKS_GALACTICA['uniprot_cytoplasm_membrane']
    df = task['gen_func']()
    print(repr(df.iloc[0]))
    print(repr(df.iloc[0]['input']))
    print(repr(df.iloc[0]['output']))
    print(df.head())
    