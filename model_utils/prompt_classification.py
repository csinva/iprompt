from typing import List

import abc
import argparse
import seaborn as sns
import datasets
import numpy as np
import os
import torch
import string
import transformers
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from . import suffix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self, model_name: str, float16=True):
        if float16:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            self.model = self.model.half()
        else:
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
        self._api_kwargs = {"model": "text-davinci-002",
                            "temperature": 0.0, "max_tokens": 1, "logprobs": 5}
        print("Initializing for calls to GPT-3 API")

    def get_logits(self, x_text: List[str]) -> torch.Tensor:
        # all negative logits
        logits = np.zeros((len(x_text), self.tokenizer.vocab_size)) - 1e4

        for i, prompt in enumerate(x_text):
            response = openai.Completion.create(
                prompt=prompt, **self._api_kwargs)
            token_logprobs = response.choices[0]['logprobs']['top_logprobs'][0].to_dict(
            )
            for token, prob in token_logprobs.items():
                token_id = self.tokenizer.encode(token)
                assert len(
                    token_id) == 1, f"token mismatch for input '{token}': {token_ids}"
                logits[i][token_id[0]] = prob  # set logits for the top 5 items

        logits = torch.tensor(logits).unsqueeze(dim=1)
        return logits.to(device).float()


def create_model(model_name: str) -> Model:
    if model_name == 'gpt3':
        return Gpt3Model()
    else:
        return Model(model_name=model_name)


def test_model_on_task_with_prefix(dset: datasets.Dataset, model: transformers.PreTrainedModel,
                                   prefix: str = '', batch_size: int = 16,
                                   restrict_to_valid_answers=True,
                                   multi_token=False,
                                   max_new_tokens=7,
                                   verbose=True,
                                   ) -> float:
    """Tests a given language model on a dataset and returns {zero,few}-shot loss. 
    Note: accuracy is computed over the set of possible answers found in the original dataset.

    Params
    ------
    dset (datasets.Dataset): dataset of examples 
    model (transformers.PreTrainedModel): language model for testing
    tokenizer (transformers.PreTrainedModel): tokenizer to accompany `model`
    prefix (str): Prefix that will be prepended to each example
    batch_size (int): batch size for evaluation
    restrict_to_valid_answers (bool):
        Whether to restrict evaluation over all tokens present in the answers.
        Only applied when multi_token is false.
    multi_token (bool):
        Whether to allow multiple tokens (uses beam search)
    max_new_tokens (int):
        number of tokens to generate when checking multi-token output

    Returns:
        loss (float): language modeling loss on examples in dataset
    """

    def get_possible_answer_mask(dataloader, model, vocab_size):
        """Compute loss only over possible answers to make task easier
        """
        possible_answer_ids = []
        for batch in dataloader:
            y_text = [answer for answer in batch['output']]
            # print(y_text)
            y_tokenized = model.tokenizer(
                y_text, return_tensors='pt', padding='longest')
            # only test on the single next token
            true_next_token_ids = y_tokenized['input_ids'][:, 0]
            possible_answer_ids.extend(true_next_token_ids.tolist())
        assert len(
            possible_answer_ids) > 0, "need multiple answers for multiple choice"

        # set up possible answers
        possible_answer_ids = torch.tensor(possible_answer_ids)
        possible_answer_mask = (
            torch.arange(start=0, end=vocab_size)[:, None]
            ==
            possible_answer_ids[None, :]
        ).any(dim=1).to(device)
        return possible_answer_mask

    # initialize
    np.random.seed(42)
    torch.manual_seed(42)
    total_loss = 0.0
    total_n = 0
    total_n_correct = 0.0
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, drop_last=False)


    for idx, batch in enumerate(dataloader):
        x_text = [(prefix + prompt) for prompt in batch['input']]
        y_text = [answer for answer in batch['output']]
        if idx == 0 and verbose:
            print('x_text[0]:' + repr(x_text[0]))
            print('y_text[0]:' + repr(y_text[0]))

        y_tokenized = model.tokenizer(
            y_text, return_tensors='pt', padding='longest').to(device)

        # note: this part currently assumes there aren't extra padding tokens at the end
        with torch.no_grad():
            ex_inputs = model.tokenizer(
                x_text, padding='longest', return_tensors='pt').to(model.model.device)

            # just decode a single token
            if not multi_token:
                # this function ensures that padded tokens are properly dealt with
                pred_next_token_logits = suffix.get_next_token_logits(
                    ex_inputs, model.model)  # note, this will break for gpt-3
                # all_token_logits = model.get_logits(x_text)
                # pred_next_token_logits = all_token_logits[:, -1, :]

                # set up mask for possible answers
                # vocab_size = model.tokenizer.vocab_size
                vocab_size = model.get_logits(['dummy text']).shape[-1]
                if restrict_to_valid_answers:
                    possible_answer_mask = get_possible_answer_mask(
                        dataloader, model, vocab_size)
                else:
                    possible_answer_mask = torch.ones(vocab_size).bool().to(device)


                # optionally take a mask over some tokens
                pred_next_token_logits = torch.where(
                    possible_answer_mask, pred_next_token_logits, torch.tensor(
                        float('-inf')).to(device)
                )

                # only test on the single next token
                true_next_token_ids = y_tokenized['input_ids'][:, 0]

                # compute loss
                loss = torch.nn.functional.nll_loss(
                    input=pred_next_token_logits, target=true_next_token_ids, reduction='sum')

                # check if correct
                total_n_correct += (pred_next_token_logits.argmax(dim=-1)
                            == true_next_token_ids.flatten()).int().sum().item()

                total_loss += loss.item()

            # deccode multiple tokens
            elif multi_token:
                bad_words = [[model.tokenizer.eos_token_id]]
                if isinstance(model.model.config.bad_words_ids, list):
                    bad_words.append(model.model.config.bad_words_ids)
                samples_t = model.model.generate(**ex_inputs,
                                                 pad_token_id=model.tokenizer.eos_token_id,
                                                 num_beams=4,
                                                 bad_words_ids=bad_words,
                                                 do_sample=False, # no randomness
                                                 max_new_tokens=max_new_tokens,
                                                 min_length=1,
                                                 length_penalty=0.6,
                                                 num_return_squences=1,
                                                #  output_scores=True,
                                                 return_dict_in_generate=True,
                                                 )
                
                samples = model.tokenizer.batch_decode(samples_t['sequences'])
                for i in range(len(samples)):
                    new_text = samples[i][len(x_text[i]):]
                    y_pred = y_text[i].rstrip(string.punctuation + string.whitespace)
                    """
                    if verbose:
                        print(i)
                        print('\tx_text', repr(x_text[i]))
                        print('\tnew_text', repr(new_text))
                        print('\ty_text', repr(y_text[i]))
                        print('\ty_pred', repr(y_pred))
                    """
                    
                    # correct if true answer is contained in the generation
                    total_n_correct += int(y_pred.strip() in new_text)

                # loss = torch.nn.functional.nll_loss(
                #     input=pred_next_token_logits, target=true_next_token_ids, reduction='sum')
                
        total_n += len(x_text)
        
    if verbose:
        print(f"Percent correct: {(total_n_correct * 100.0 / total_n):.2f}")
    if not multi_token:
        return (total_loss / total_n), (total_n_correct * 100.0 / total_n)
    else:
        return np.nan, (total_n_correct * 100.0 / total_n)
