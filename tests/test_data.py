import os
from os.path import dirname

def test_data():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    exit_value = os.system('python ' + os.path.join(repo_dir, 'data.py '))
    assert exit_value == 0, 'default data loading passed'