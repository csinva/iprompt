from typing import Dict

import functools

import datasets
import pandas as pd


SST2_SPLIT_DICT = {
    "sst2_train": "train",
    "sst2_validation": "validation",
    "sst2_test": "test",
}

IMDB_SPLIT_DICT = {
    "imdb_train": "train",
    "imdb_test": "test",
}

RT_SPLIT_DICT = {
    "rt_train": "train",
    "rt_validation": "validation",
    "rt_test": "test",
}

FFB_SPLIT_DICT = {
    "ffb_train": "train",
}

TWEETS_SPLIT_DICT = {
    "tweets_train": "train"
}

ALL_SPLIT_DICT = {**SST2_SPLIT_DICT, **IMDB_SPLIT_DICT, **RT_SPLIT_DICT, **FFB_SPLIT_DICT, **TWEETS_SPLIT_DICT}

"""
    # process dset
    if args.dataset == 'sst2':
        del dataset['test']
        args.dataset_key_text = 'sentence'
    if args.dataset == 'financial_phrasebank':
        args.dataset_key_text = 'sentence'        
    elif args.dataset == 'imdb':
        del dataset['unsupervised']
        dataset['validation'] = dataset['test']
        del dataset['test']
        args.dataset_key_text = 'text'
    elif args.dataset == 'emotion':
        del dataset['test']
        args.dataset_key_text = 'text'
    elif args.dataset == 'rotten_tomatoes':
        del dataset['test']
        args.dataset_key_text = 'text'       
    elif args.dataset == 'tweet_eval':
        del dataset['test']
        args.dataset_key_text = 'text'               
    #if args.subsample > 0:
    #    dataset['train'] = dataset['train'].select(range(args.subsample))
    return dataset, args
"""

LABEL_MAP = {
    # neg, pos
    "rotten_tomatoes": { 0: "No", 1: "Yes" },
    "sst2": { 0: "No", 1: "Yes" },
    "imdb": { 0: "No", 1: "Yes" },
    # neg, neutral, pos
    "financial_phrasebank": { 0: "No", 1: "Maybe", 2: "Yes" },
    # not hate speech, yes hate speech
    "tweets_hate_speech_detection": { 0: "No", 1: "Yes" },
}


def make_row_sentiment(row: Dict[str, str], dataset_name: str, text_key: str) -> Dict[str, str]:
    text_input = f'Input: {row[text_key]} Answer:'
    sentiment = LABEL_MAP[dataset_name][row['label']]
    text_output =  f' {sentiment}\n'
    return {
        "input": text_input,
        "output": text_output,
        "text": (text_input + text_output + '\n\n'),
    }


def fetch_classification_data(dataset_split: str, dataset_name: str, text_key: str) -> pd.DataFrame:
    print("**loading data:", dataset_name, "//", ALL_SPLIT_DICT[dataset_split])
    if dataset_name == 'financial_phrasebank':
        raw_dataset = datasets.load_dataset(dataset_name, 'sentences_allagree', split=ALL_SPLIT_DICT[dataset_split])
    else:
        raw_dataset = datasets.load_dataset(dataset_name, split=ALL_SPLIT_DICT[dataset_split])
    raw_dataset = raw_dataset.filter(lambda row: row["label"] in LABEL_MAP[dataset_name])
    dataset = raw_dataset.map(functools.partial(make_row_sentiment, dataset_name=dataset_name, text_key=text_key))
    return dataset.to_pandas()


SENTIMENT_CHECK_ANSWER_FUNC = r'positive|good|happy|negative|bad|sad'
HATESPEECH_DESCRIPTION = 'Answer Yes if the input is hate speech and No otherwise.'
SENTIMENT_DESCRIPTION = 'Answer Yes if the input is positive and No if the input is negative.'
SENTIMENT_DESCRIPTION_FFB =  'Answer Yes for positive, No for negative, and Maybe for neutral.'
SENTIMENT_SUFFIX = 'Answer "positive" or "negative" depending on the'


TASKS_CLASSIFICATION = {}
TASKS_CLASSIFICATION['SUFFIXES'] = {}

# SST2
for split_name in SST2_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': SENTIMENT_DESCRIPTION,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="sst2", text_key="sentence"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

# IMDB
for split_name in IMDB_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': SENTIMENT_DESCRIPTION,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="imdb", text_key="text"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

# RT
for split_name in RT_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': SENTIMENT_DESCRIPTION,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="rotten_tomatoes", text_key="text"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

# FFB
for split_name in FFB_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': SENTIMENT_DESCRIPTION_FFB,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="financial_phrasebank", text_key="sentence"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

# TWEET
for split_name in TWEETS_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': HATESPEECH_DESCRIPTION,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="tweets_hate_speech_detection", text_key="tweet"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

if __name__ == '__main__':
    for split_name in TASKS_CLASSIFICATION.keys():
        if split_name == 'SUFFIXES': continue
        TASKS_CLASSIFICATION[split_name]['gen_func'](split_name)
    print(TASKS_CLASSIFICATION)
