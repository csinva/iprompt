from typing import List, Tuple

import argparse
import glob
import os
import re

import pandas as pd


def get_file_number(filename: str) -> int:
    return int(re.search(r'prefix_(\d+).p', filename).group(1))


def read_dfs(folder_name: str) -> Tuple[List[int], List[pd.DataFrame]]:
    assert os.path.exists(folder_name)
    files = glob.glob(os.path.join(folder_name, 'prefix_*.p'))
    files.sort(key=get_file_number) # sort files numerically
    assert len(files) > 0, 'no prefix files found with format prefix_\{step\}.p'
    dfs = []
    for filename in files:
        dfs.append(pd.read_pickle(filename))
    return list(map(get_file_number, files)), dfs

def create_prefix_data(folder_name: str):
    """creates output CSV from inputs.

    we want to output a CSV file with prefixes as rows
    and accuracy as columns.
    """
    steps, dfs = read_dfs(folder_name=folder_name)
    #
    # [1/3] Iterate through and get all prefixes
    #
    all_prefixes = set()
    for step, df in zip(steps, dfs):
        all_prefixes.update(set(df['prefix']))
    #
    # [2/3] Create output df
    #
    all_prefixes = list(all_prefixes)
    output_data = []
    for step, df in zip(steps, dfs):
        step_data = []
        acc_by_prefix = dict(zip(df['prefix'], df['accuracy']))
        for prefix in all_prefixes:
            step_data.append(acc_by_prefix.get(prefix, 0.0))
        output_data.append(step_data)
    # [3/3] Output data to file
    df = pd.DataFrame(output_data, columns=all_prefixes)
    df = df.transpose()
    bar_chart_outfile = os.path.join(folder_name, 'bar_chart.csv')
    df.to_csv(bar_chart_outfile)
    print(f'wrote {len(steps)} steps and {len(all_prefixes)} prefixes to {bar_chart_outfile}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name', type=str,
                        help='folder name with prefix outputs')
    args = parser.parse_args()
    create_prefix_data(folder_name=args.folder_name)

if __name__ == '__main__':
    main()