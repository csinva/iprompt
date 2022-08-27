import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datasets import Dataset
import data
import pickle as pkl


def train(dset, model, tokenizer, batch_size=100):
    np.random.seed(13)
    torch.manual_seed(13)
    device = 'cuda'

    r = defaultdict(list)

    model = model.to(device)
    trans = model._modules['transformer']
    wte = trans.wte.to(device)

    # initialize prefix
    prefix_str = ["x the following two numbers: "]
    prefix_inputs = tokenizer(prefix_str, return_tensors="pt").to(device)
    prefix_emb = wte.forward(prefix_inputs['input_ids'])
    prefix_emb = torch.nn.Parameter(prefix_emb).to(device)

    # optimizer
    optim = torch.optim.Adam([prefix_emb])

    for epoch in tqdm(range(100)):

        # embedding for example
        ex_num = 0
        batch_size = batch_size
        ex = dset[ex_num: ex_num + batch_size]
        x_text = ex['input']
        y_text = ex['output']
        full_text = [x_text[i] + y_text[i] for i in range(len(x_text))]
        ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
        ex_embs = wte.forward(ex_inputs['input_ids'].to(
            device)).to(device)  # this is the key param

        # concatenate prefix + example
        emb = torch.cat((prefix_emb.repeat(batch_size, 1, 1),
                        ex_embs), dim=1)

        # go through model
        outputs = model(inputs_embeds=emb)

        # calculate loss
        idxs_correct = tokenizer(y_text, return_tensors='pt')['input_ids']
        assert idxs_correct.nelement() == batch_size, 'For now assume that answer is a single token'
        y_idx_correct = idxs_correct[0]
        # (batch_size, seq_len, vocab_size)
        logit_answer = outputs['logits'][0, -1, y_idx_correct]

        # optimize
        optim.zero_grad()
        loss = -1 * logit_answer
        loss.backward()
        optim.step()

        # save stuff
        r['embs'].append(prefix_emb.detach().cpu().numpy())
        r['losses'].append(loss.item())
        # print('losses', loss)
    return r


if __name__ == '__main__':
    checkpoint = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, output_hidden_states=True)
    dset = data.get_data(N=1000)
    r = train(dset, model, tokenizer, batch_size=500)

    pkl.dump(r, open('results.pkl', 'wb'))
