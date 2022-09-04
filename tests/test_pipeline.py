import os
from os.path import dirname

def test_pipeline():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value = os.system('python ' + os.path.join(repo_dir, '01_train.py '))
    assert exit_value == 0, 'default pipeline passed'