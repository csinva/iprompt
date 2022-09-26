from typing import Dict

import datasets
import pandas as pd


SPLIT_DICT = {
    "sst2_train": "train",
    "sst2_validation": "validation",
    "sst2_test": "test",
}
SENTIMENT_STR = {
    0: "negative", 1: "positive"
}

def make_row_sst2(row: Dict[str, str]) -> Dict[str, str]:
    text_input = f'Input: {row["sentence"]} Answer:'
    sentiment = SENTIMENT_STR[row['label']]
    text_output =  f' {sentiment}\n'
    return {
        "input": text_input,
        "output": text_output,
        "text": (text_input + text_output + '\n\n'),
    }


def fetch_sst2_data(dataset_split: str) -> pd.DataFrame:
    assert dataset_split in SPLIT_DICT
    raw_dataset = datasets.load_dataset("sst2", split=SPLIT_DICT[dataset_split])
    dataset = raw_dataset.map(make_row_sst2)
    return dataset.to_pandas()


SST2_CHECK_ANSWER_FUNC = r'positive|good|happy|negative|bad|sad'
SST2_DESCRIPTION = 'Answer "positive" or "negative" depending on the sentiment of the input."'


TASKS_SST2 = {}
TASKS_SST2['SUFFIXES'] = {}
for split_name in SPLIT_DICT.keys():
    TASKS_SST2[split_name] = {
        'check_answer_func': SST2_CHECK_ANSWER_FUNC,
        'description': SST2_DESCRIPTION,
        'gen_func': fetch_sst2_data
    }
    TASKS_SST2['SUFFIXES'][split_name] = 'Answer "positive" or "negative" depending on the'

if __name__ == '__main__':
    fetch_sst2_data('sst2_train')
    print(TASKS_SST2)
