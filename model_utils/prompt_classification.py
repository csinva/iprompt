import argparse
import seaborn as sns
import datasets
import numpy as np
import torch
import transformers
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt

def test_model_on_task_with_prefix(dset: datasets.Dataset, model: transformers.PreTrainedModel,
                                   tokenizer: transformers.PreTrainedTokenizer,
                                   prefix: str = '', batch_size: int = 32, device='cpu') -> float:
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

    model.eval()
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_n = 0
    total_n_correct = 0.0

    # Compute loss only over possible answers to make task easier
    possible_answer_ids = []
    for batch in dataloader:
        y_text = [answer for answer in batch['output']]
        # print(y_text)
        y_tokenized = tokenizer(y_text, return_tensors='pt', padding='longest')
        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]
        possible_answer_ids.extend(true_next_token_ids.tolist())

    assert len(
        possible_answer_ids) > 0, "need multiple answers for multiple choice"

    possible_answer_ids = torch.tensor(possible_answer_ids)
    vocab_size = len(tokenizer.vocab)
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

        x_tokenized = tokenizer(
            x_text, return_tensors='pt', padding='longest').to(device)
        y_tokenized = tokenizer(
            y_text, return_tensors='pt', padding='longest').to(device)

        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]

        with torch.no_grad():
            all_token_logits = model(**x_tokenized)['logits']
            pred_next_token_logits = all_token_logits[:, -1, :]
            # import pdb; pdb.set_trace()
            pred_next_token_logits = torch.where(
                possible_answer_mask, pred_next_token_logits, torch.tensor(
                    float('-inf')).to(device)
            )
            loss = torch.nn.functional.cross_entropy(
                input=pred_next_token_logits, target=true_next_token_ids, reduction='sum')
        total_loss += loss.item()
        total_n += len(x_text)
        total_n_correct += (pred_next_token_logits.argmax(dim=-1)
                            == true_next_token_ids.flatten()).int().sum().item()

    return (total_loss / total_n), (total_n_correct * 100.0 / total_n)
