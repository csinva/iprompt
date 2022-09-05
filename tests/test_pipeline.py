import os
from os.path import dirname


def test_pipeline():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value=os.system('python ' + os.path.join(repo_dir, '01_train.py --use_cache 0'))
    assert exit_value == 0, 'default pipeline passed'


def test_nli_single_query():
    repo_dir=dirname(dirname(os.path.abspath(__file__)))
    exit_value=os.system('python ' + os.path.join(repo_dir, '01_train.py ' +
        '--task_name task1147_country_currency --use_single_query 1 --use_cache 0'))
    assert exit_value == 0, 'nli single-query pipeline passed'
