import torch
import torch.distributed as dist
from parallelformers import parallelize
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          pipeline, top_k_top_p_filtering)

device_count = torch.cuda.device_count()
device = 'cpu' if device_count == 0 else 'cuda'


def model_to_device(model):
    if device_count == 0:
        return model
    elif device_count == 1:
        return model.to(device)
    elif device_count > 1:
        parallelize(model, num_gpus=device_count, fp16=False, verbose='detail')
        # print memory states
        print(model.memory_allocated())
        print(model.memory_reserved())
        return model


def inputs_to_device(inputs):
    if device_count == 0 or device_count > 1:
        return inputs
    elif device_count == 1:
        return inputs.to(device)


if __name__ == '__main__':
    """Run this to check whether parallelformers works on multi-gpu setup
    """
    print('loading model...')
    for checkpoint in ['gpt2-medium', 'gpt2-large', 'gpt2-xl',
                       'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neox-20b']:
        print('checking', checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, output_hidden_states=True)

        # parallelize the model
        parallelize(model, num_gpus=1, fp16=False, verbose='detail')

        # prepare inputs
        raw_inputs = ["1+3=4"]
        inputs = tokenizer(raw_inputs, return_tensors="pt")

        # predict
        outputs = model(**inputs)
        assert outputs is not None, f'outputs threw an error for {checkpoint}'
