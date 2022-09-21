from typing import List

import abc
import argparse
import seaborn as sns
import datasets
import numpy as np
import os
import torch
import transformers
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self, model_name: str):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.model.to(device)
    
    def get_logits(self, x_text: List[str]) -> torch.Tensor:
        x_tokenized = self.tokenizer(
            x_text, return_tensors='pt', padding='longest'
        ).to(device)
        return self.model(**x_tokenized)['logits'].log_softmax(dim=1)


class Gpt3Model(Model):
    def __init__(self):
        assert 'OPENAI_API_KEY' in os.environ, 'need to set OPENAI_API_KEY in env to use GPT-3 API'
        import openai
        openai.api_key = os.environ['OPENAI_API_KEY']
        self._num_requests = 0
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._api_kwargs = { "model": "text-davinci-002", "temperature": 0.0, "max_tokens": 1, "logprobs": 5 }
        print("Initializing for calls to GPT-3 API")
    
    def get_logits(self, x_text: List[str]) -> torch.Tensor:
        # all negative logits
        logits = np.zeros((len(x_text), self.tokenizer.vocab_size)) - 1e4

        for i, prompt in enumerate(x_text):
            response = openai.Completion.create(prompt=prompt, **self._api_kwargs)
            token_logprobs = response.choices[0]['logprobs']['top_logprobs'][0].to_dict()
            for token, prob in token_logprobs.items():
                token_id = self.tokenizer.encode(token)
                assert len(token_id) == 1, f"token mismatch for input '{token}': {token_ids}"
                logits[i][token_id[0]] = prob # set logits for the top 5 items
        
        logits = torch.tensor(logits).unsqueeze(dim=1)
        return logits.to(device).float()


def create_model(model_name: str) -> Model:
    if model_name == 'gpt3':
        return Gpt3Model()
    else:
        return Model(model_name=model_name)

def test_model_on_task_with_prefix(dset: datasets.Dataset, model: transformers.PreTrainedModel,
                                   prefix: str = '', batch_size: int = 16) -> float:
    """Tests a given language model on a dataset and returns {zero,few}-shot loss. 

    Args:
        dset (datasets.Dataset): dataset of examples 
        model (transformers.PreTrainedModel): language model for testing
        tokenizer (transformers.PreTrainedModel): tokenizer to accompany `model`
        prefix (str): Prefix that will be prepended to each example
        batch_size (int): batch size for evaluation

    Returns:
        loss (float): language modeling loss on examples in dataset
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # set up dataloader
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, drop_last=False)
    

    # Compute loss only over possible answers to make task easier
    total_loss = 0.0
    total_n = 0
    total_n_correct = 0.0
    possible_answer_ids = []
    for batch in dataloader:
        y_text = [answer for answer in batch['output']]
        # print(y_text)
        y_tokenized = model.tokenizer(y_text, return_tensors='pt', padding='longest')
        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]
        possible_answer_ids.extend(true_next_token_ids.tolist())
    assert len(
        possible_answer_ids) > 0, "need multiple answers for multiple choice"


    # set up possible answers
    possible_answer_ids = torch.tensor(possible_answer_ids)
    # vocab_size = model.tokenizer.vocab_size
    vocab_size = model.get_logits(['dummy text']).shape[-1]
    possible_answer_mask = (
        torch.arange(start=0, end=vocab_size)[:, None]
        ==
        possible_answer_ids[None, :]
    ).any(dim=1).to(device)

    for idx, batch in enumerate(dataloader):
        x_text = [(prefix + prompt) for prompt in batch['input']]
        y_text = [answer for answer in batch['output']]
        if idx == 0:
            print(list(zip(x_text[:2], y_text[:2])))

        y_tokenized = model.tokenizer(
            y_text, return_tensors='pt', padding='longest').to(device)

        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]


        with torch.no_grad():
            all_token_logits = model.get_logits(x_text)
            pred_next_token_logits = all_token_logits[:, -1, :]
            pred_next_token_logits = torch.where(
                possible_answer_mask, pred_next_token_logits, torch.tensor(
                    float('-inf')).to(device)
            )
            loss = torch.nn.functional.nll_loss(
                input=pred_next_token_logits, target=true_next_token_ids, reduction='sum')
            
        total_loss += loss.item()
        total_n += len(x_text)
        total_n_correct += (pred_next_token_logits.argmax(dim=-1)
                            == true_next_token_ids.flatten()).int().sum().item()


    print(f"Percent correct: {(total_n_correct * 100.0 / total_n):.2f}")
    return (total_loss / total_n), (total_n_correct * 100.0 / total_n)
