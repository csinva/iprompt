from typing import Any, Dict

import argparse
import io
import os

import datasets
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import transformers
import pandas as pd

import iprompt.data as data
from iprompt.prefix import AutoPrompt, iPrompt
from iprompt.prefix.utils import get_preprefix_from_args, load_lm_from_checkpoint


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def rerank_dict(json_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recomputes acc & loss on test data and re-ranks prefixes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = argparse.Namespace(**json_dict)
    preprefix = get_preprefix_from_args(args=args)
    dset, check_answer_func, description = data.get_data(
        task_name=args.task_name, n_shots=args.n_shots, train_split_frac=args.train_split_frac, max_dset_size=args.max_dset_size,
        template_num_task_phrasing=args.template_num_task_phrasing, max_digit=args.max_digit
    )
    dset_train, _dset_test = dset
    
    if args.mask_possible_answers:
        vocab_size = len(tokenizer.vocab)
        possible_answer_mask = (
            torch.arange(start=0, end=vocab_size)[:, None]
            ==
            possible_answer_ids[None, :]
        ).any(dim=1).to(device)
    else:
        possible_answer_mask = None

    model_cls_dict = {
        'autoprompt': AutoPrompt,
        'genetic': iPrompt,  # outdated alias
        'iprompt': iPrompt,
        'suff': iPrompt, # also fixes some hyperparams to specific values
    }
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.eos_token = tokenizer.eos_token or 0
    tokenizer.pad_token = tokenizer.eos_token

    lm = load_lm_from_checkpoint(
        checkpoint=args.checkpoint, float16=args.llm_float16).to(device)
    model = model_cls_dict[args.model_cls](
        args=args,
        loss_func=None,
        model=lm,
        tokenizer=tokenizer,
        preprefix=preprefix
    )
    n_eval = 64
    eval_dset = datasets.Dataset.from_dict(dset_train[:n_eval])
    eval_dataloader = DataLoader(
        eval_dset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # fake add all prefixes.
    for prefix_ids in json_dict['prefix_ids']:
         model._prefix_pool.update(
            prefix=torch.tensor(prefix_ids),
            loss=torch.tensor(0.0),
            accuracy=torch.tensor(0.0),
        )

    json_dict.update(model.serialize(eval_dataloader, possible_answer_mask))
    json_dict["prefixes__check_answer_func"] = list(
        map(check_answer_func, json_dict["prefixes"]))
    
    return json_dict


def rerank_folder(folder_name: str):
    pickle_filename = os.path.join(folder_name, 'results.pkl')
    out_file = os.path.join(folder_name, 'results_reranked.pkl')

    if os.path.exists(out_file):
        print(f'Exiting: file already exists {out_file}')
        exit()

    if not os.path.exists(pickle_filename):
        raise FileNotFoundException(f'No results file found at {pickle_filename}')
    json_dict = CPU_Unpickler(open(pickle_filename, 'rb')).load()

    # set old args
    json_dict["iprompt_do_final_reranking"] = json_dict.get("iprompt_do_final_reranking", 1)
    json_dict["iprompt_criterion"] = json_dict.get("iprompt_criterion", "loss")

    new_json_dict = rerank_dict(json_dict)
    out_file = os.path.join(folder_name, 'results_reranked.pkl')
    pkl.dump(new_json_dict, open(out_file, 'wb'))
    print('wrote to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name")
    args = parser.parse_args()
    rerank_folder(folder_name=args.folder_name)