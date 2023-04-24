from typing import Dict

import functools

import datasets
import pandas as pd


SST2_SPLIT_DICT = {
    "sst2_train": "train",
    "sst2_test": "validation",
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
    "ffb_test": "train", # special case (see DATASETS_MISSING_TEST_SET)
}

TWEETS_SPLIT_DICT = {
    "tweets_train": "train",
    "tweets_test": "test", # special case (see DATASETS_MISSING_TEST_SET)
}

EMOTION_SPLIT_DICT = {
    "emotion_train": "train",
    "emotion_test": "test",
}


DATASETS_MISSING_TEST_SET = {'financial_phrasebank'}


ALL_SPLIT_DICT = {**SST2_SPLIT_DICT, **IMDB_SPLIT_DICT, **RT_SPLIT_DICT, **FFB_SPLIT_DICT, **TWEETS_SPLIT_DICT, **EMOTION_SPLIT_DICT}

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

# LABEL_MAP = {
#     # neg, pos
#     "rotten_tomatoes": { 0: "No", 1: "Yes" },
#     "sst2": { 0: "No", 1: "Yes" },
#     "imdb": { 0: "No", 1: "Yes" },
#     # neg, neutral, pos
#     "financial_phrasebank": { 0: "No", 1: "Maybe", 2: "Yes" },
#     # not hate speech, yes hate speech
#     "tweets_hate_speech": { 0: "No", 1: "Yes" },
# }

LABEL_MAP = {
    # neg, pos
    "rotten_tomatoes": { 0: "negative", 1: "positive" },
    "sst2": { 0: "negative", 1: "positive" },
    "imdb": { 0: "negative", 1: "positive" },
    # neg, neutral, pos
    "financial_phrasebank": { 0: "negative", 1: "neutral", 2: "positive" },
    # not hate speech, yes hate speech
    "tweets_hate_speech": { 0: "negative", 1: "positive" },
    'emotion': {0: 'Sad', 1: 'Happ', 2: 'Love', 3: 'Ang', 4: 'Fear', 5: 'Surpris'},
}



# initial_str = ""
# initial_str = "Movie Review: "
initial_str = "Input: "
output_str = " Output:"
def make_row_sentiment_func(no_quotes: bool):
    no_quotes = True
    def make_row_sentiment__(row: Dict[str, str], dataset_name: str, text_key: str) -> Dict[str, str]:
        if no_quotes:
            text_input = f'{initial_str}{row[text_key].strip()} {output_str}'
        else:
            text_input = f'{initial_str}"{row[text_key].strip()}" {output_str}'
        sentiment = LABEL_MAP[dataset_name][row['label']]
        text_output =  f' {sentiment}\n'
        return {
            "input": text_input,
            "output": text_output,
            "text": (text_input + text_output + '\n\n'),
        }
    return make_row_sentiment__


def fetch_classification_data(dataset_split: str, dataset_name: str, text_key: str) -> pd.DataFrame:
    no_quotes = True
    print("**loading data:", dataset_name, "//", ALL_SPLIT_DICT[dataset_split])
    # load dataset
    if dataset_name == 'financial_phrasebank':
        raw_dataset = datasets.load_dataset(dataset_name, 'sentences_allagree', split=ALL_SPLIT_DICT[dataset_split])
    elif dataset_name == 'tweets_hate_speech':
        raw_dataset = datasets.load_dataset('tweet_eval', 'hate', split=ALL_SPLIT_DICT[dataset_split])
    else:
        raw_dataset = datasets.load_dataset(dataset_name, split=ALL_SPLIT_DICT[dataset_split])
    raw_dataset = raw_dataset.shuffle(seed=2) # shuffle for label-matching
    # make train-test split, in special cases
    if dataset_name in DATASETS_MISSING_TEST_SET:
        __N = round(len(raw_dataset) * 0.75)
        if ALL_SPLIT_DICT[dataset_split] == 'train':
            # take first 75%
            raw_dataset = datasets.Dataset.from_dict(raw_dataset[:__N])
        elif ALL_SPLIT_DICT[dataset_split] == 'test':
            # take last 25%
            raw_dataset = datasets.Dataset.from_dict(raw_dataset[__N:])
        else:
            raise ValueError(f'unknown dataset split {dataset_split} for dataset {dataset_name}')
    # get labels
    raw_dataset = raw_dataset.filter(lambda row: row["label"] in LABEL_MAP[dataset_name])
    if not len(raw_dataset): raise ValueError("got no datapoints after filtering for valid labels")
    # make rows
    dataset = raw_dataset.map(
        functools.partial(make_row_sentiment_func(no_quotes=no_quotes), dataset_name=dataset_name, text_key=text_key)
    )
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
        'gen_func': functools.partial(fetch_classification_data, dataset_name="tweets_hate_speech", text_key="text"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

# EMOTION
for split_name in EMOTION_SPLIT_DICT.keys():
    TASKS_CLASSIFICATION[split_name] = {
        'check_answer_func': SENTIMENT_CHECK_ANSWER_FUNC,
        'description': SENTIMENT_DESCRIPTION,
        'gen_func': functools.partial(fetch_classification_data, dataset_name="emotion", text_key="text"),
    }
    TASKS_CLASSIFICATION['SUFFIXES'][split_name] = SENTIMENT_SUFFIX

if __name__ == '__main__':
    for split_name in TASKS_CLASSIFICATION.keys():
        if split_name == 'SUFFIXES': continue
        print(split_name)
        print(TASKS_CLASSIFICATION[split_name]['gen_func'](split_name))
    print(TASKS_CLASSIFICATION)
